from transformers import Trainer
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import is_peft_available
from peft import PeftModel

import inspect

def _is_peft_model(model):
    return is_peft_available() and isinstance(model, PeftModel)

if is_peft_available():
    from peft import PeftModel


class VLATrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        log_loss = outputs['log_loss'] if 'log_loss' in outputs else None
        if log_loss is not None:
            self.log(log_loss)


        # log loss
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self._call_label_smoother(outputs, labels, shift_labels=True, num_items_in_batch=num_items_in_batch)
            else:
                loss = self._call_label_smoother(outputs, labels, shift_labels=False, num_items_in_batch=num_items_in_batch)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def _call_label_smoother(self, outputs, labels, shift_labels: bool, num_items_in_batch=None):
        # Transformers versions differ on whether LabelSmoother supports num_items_in_batch.
        try:
            params = inspect.signature(self.label_smoother).parameters
        except (TypeError, ValueError):
            params = {}

        if "num_items_in_batch" in params:
            return self.label_smoother(
                outputs,
                labels,
                shift_labels=shift_labels,
                num_items_in_batch=num_items_in_batch,
            )
        return self.label_smoother(outputs, labels, shift_labels=shift_labels)
