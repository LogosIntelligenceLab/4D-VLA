
# # # 1. openvla
# # cd /SSD_DISK/users/zhangjiahui/openvla
# # export data_root_dit="/SSD_DISK/users/zhangjiahui/openvla/data/franka/openvla_data_task2"
# # export max_steps=200
# # export save_steps=200
# # bash vla-scripts/finetune.sh


# cd /SSD_DISK/users/zhangjiahui/openvla/internvl_chat
# # 2. baseline
# export output_dit="work_dirs_ssd_group/franka_new_bs32_20ep_baseline_50samples_v5_norm_repeatimg_task2"
# export meta_path="shell/exp_data/franka_task2_50samples_hist1.json"
# bash shell/exp_shell/franka/franka_new_bs32_20ep_baseline_50samples_v5.sh

# # 3. baseline + pt
# export output_dit="work_dirs_ssd_group/franka_new_bs32_20ep_baseline_pt_50samples_v5_norm_task2"
# export meta_path="shell/exp_data/franka_task2_50samples_hist1.json"
# bash shell/exp_shell/franka/franka_new_bs32_20ep_baseline_pt_50samples_v5.sh

# # # 4. baseline + pt + dep
# export output_dit="work_dirs_ssd_group/franka_new_bs32_20ep_baseline_pt_dep_50samples_v5_norm_no_crop_task2_aug"
# export meta_path="shell/exp_data/franka_task2_50samples_hist1.json"
# bash shell/exp_shell/franka/franka_new_bs32_20ep_baseline_pt_dep_50samples_v5.sh

# # # 5. baseline + pt + mb
# export output_dit="work_dirs_ssd_group/franka_new_bs32_20ep_baseline_pt_mb_50samples_v5_norm_task2_aug"
# export meta_path="shell/exp_data/franka_task2_50samples_mb5.json"
# bash shell/exp_shell/franka/franka_new_bs32_20ep_baseline_pt_mb_50samples_v5.sh

# 6. baseline + pt + mb + dep
# export output_dir="work_dirs_ssd_group/franka_new_bs32_40ep_baseline_pt_mb_aug_dep_50samples_v5_norm_crop_unfrozen_embed_40ep_task2_new_drop08_80ep"
# export output_dir="work_dirs_ssd_group/franka_new_bs32_40ep_baseline_pt_mb_aug_dep_50samples_v5_norm_crop_unfrozen_embed_40ep_task2_new_drop08_60ep"
# export output_dir="work_dirs_ssd_group/franka_new_bs32_40ep_baseline_pt_mb_aug_dep_50samples_v5_norm_crop_unfrozen_embed_40ep_task2_new_drop08_40ep_no_crop"
# export meta_path="shell/exp_data/franka_task2_50samples_mb5.json"
# bash shell/exp_shell/franka/franka_new_bs32_20ep_baseline_pt_mb_aug_dep_50samples_v5.sh



# # 1. openvla
# cd /SSD_DISK/users/zhangjiahui/openvla
# export data_root_dit="/SSD_DISK/users/zhangjiahui/openvla/data/franka/openvla_data_task3"
# export max_steps=500
# export save_steps=500
# bash vla-scripts/finetune.sh


# cd /SSD_DISK/users/zhangjiahui/openvla/internvl_chat
# # 2. baseline
# export output_dit="work_dirs_ssd_group/franka_new_bs32_20ep_baseline_50samples_v5_norm_repeatimg_task3"
# export meta_path="shell/exp_data/franka_task3_50samples_hist1.json"
# bash shell/exp_shell/franka/franka_new_bs32_20ep_baseline_50samples_v5.sh

# # 3. baseline + pt
# export output_dit="work_dirs_ssd_group/franka_new_bs32_20ep_baseline_pt_50samples_v5_norm_task3"
# export meta_path="shell/exp_data/franka_task3_50samples_hist1.json"
# bash shell/exp_shell/franka/franka_new_bs32_20ep_baseline_pt_50samples_v5.sh

# # # 4. baseline + pt + dep
# export output_dit="work_dirs_ssd_group/franka_new_bs32_20ep_baseline_pt_dep_50samples_v5_norm_no_crop_task3_aug"
# export meta_path="shell/exp_data/franka_task3_50samples_hist1.json"
# bash shell/exp_shell/franka/franka_new_bs32_20ep_baseline_pt_dep_50samples_v5.sh

# # 5. baseline + pt + mb
# export output_dit="work_dirs_ssd_group/franka_new_bs32_20ep_baseline_pt_mb_50samples_v5_norm_task3_aug"
# export meta_path="shell/exp_data/franka_task3_50samples_mb5.json"
# bash shell/exp_shell/franka/franka_new_bs32_20ep_baseline_pt_mb_50samples_v5.sh

# 6. baseline + pt + mb + dep
# # export output_dir="work_dirs_ssd_group/franka_new_bs32_40ep_baseline_pt_mb_aug_dep_50samples_v5_norm_crop_unfrozen_embed_40ep_task3_new_drop08_40ep_nocrop"
# export output_dir="work_dirs_ssd_group/franka_new_bs32_40ep_baseline_pt_mb_aug_dep_50samples_v5_norm_crop_unfrozen_embed_40ep_task3_new_drop08_60ep"
# export meta_path="shell/exp_data/franka_task3_50samples_mb5.json"
# bash shell/exp_shell/franka/franka_new_bs32_20ep_baseline_pt_mb_aug_dep_50samples_v5.sh


# # 1. openvla
# cd /SSD_DISK/users/zhangjiahui/openvla
# export data_root_dit="/SSD_DISK/users/zhangjiahui/openvla/data/franka/openvla_data_task4"
# export max_steps=1000
# export save_steps=1000
# bash vla-scripts/finetune.sh


# # cd /SSD_DISK/users/zhangjiahui/openvla/internvl_chat
# # 2. baseline
# export output_dit="work_dirs_ssd_group/franka_new_bs32_20ep_baseline_50samples_v5_norm_repeatimg_task4"
# export meta_path="shell/exp_data/franka_task4_50samples_hist1.json"
# bash shell/exp_shell/franka/franka_new_bs32_20ep_baseline_50samples_v5.sh

# # 3. baseline + pt
# export output_dit="work_dirs_ssd_group/franka_new_bs32_20ep_baseline_pt_50samples_v5_norm_task4"
# export meta_path="shell/exp_data/franka_task4_50samples_hist1.json"
# bash shell/exp_shell/franka/franka_new_bs32_20ep_baseline_pt_50samples_v5.sh

# # # 4. baseline + pt + dep
# export output_dit="work_dirs_ssd_group/franka_new_bs32_20ep_baseline_pt_dep_50samples_v5_norm_no_crop_task4_aug"
# export meta_path="shell/exp_data/franka_task4_50samples_hist1.json"
# bash shell/exp_shell/franka/franka_new_bs32_20ep_baseline_pt_dep_50samples_v5.sh

# # 5. baseline + pt + mb
# export output_dit="work_dirs_ssd_group/franka_new_bs32_20ep_baseline_pt_mb_50samples_v5_norm_task4_aug"
# export meta_path="shell/exp_data/franka_task4_50samples_mb5.json"
# bash shell/exp_shell/franka/franka_new_bs32_20ep_baseline_pt_mb_50samples_v5.sh

# # 6. baseline + pt + mb + dep
# export output_dir="work_dirs_ssd_group/franka_new_bs32_40ep_baseline_pt_mb_aug_dep_50samples_v5_norm_crop_unfrozen_embed_40ep_task4_new_drop08_50ep"
# export meta_path="shell/exp_data/franka_task4_50samples_mb5.json"
# bash shell/exp_shell/franka/franka_new_bs32_20ep_baseline_pt_mb_aug_dep_50samples_v5.sh



# # 1. openvla
# cd /SSD_DISK/users/zhangjiahui/openvla
# export data_root_dit="/SSD_DISK/users/zhangjiahui/openvla/data/franka/openvla_data_task5"
# export max_steps=1000
# export save_steps=1000
# bash vla-scripts/finetune.sh


# # cd /SSD_DISK/users/zhangjiahui/openvla/internvl_chat
# # 2. baseline
# export output_dit="work_dirs_ssd_group/franka_new_bs32_20ep_baseline_50samples_v5_norm_repeatimg_task5"
# export meta_path="shell/exp_data/franka_task5_50samples_hist1.json"
# bash shell/exp_shell/franka/franka_new_bs32_20ep_baseline_50samples_v5.sh

# # 3. baseline + pt
# export output_dit="work_dirs_ssd_group/franka_new_bs32_20ep_baseline_pt_50samples_v5_norm_task5"
# export meta_path="shell/exp_data/franka_task5_50samples_hist1.json"
# bash shell/exp_shell/franka/franka_new_bs32_20ep_baseline_pt_50samples_v5.sh

# # # 4. baseline + pt + dep
# export output_dit="work_dirs_ssd_group/franka_new_bs32_20ep_baseline_pt_dep_50samples_v5_norm_no_crop_task5_aug"
# export meta_path="shell/exp_data/franka_task5_50samples_hist1.json"
# bash shell/exp_shell/franka/franka_new_bs32_20ep_baseline_pt_dep_50samples_v5.sh

# # 5. baseline + pt + mb
# export output_dit="work_dirs_ssd_group/franka_new_bs32_20ep_baseline_pt_mb_50samples_v5_norm_task5_aug_drop08_30ep"
# export meta_path="shell/exp_data/franka_task5_50samples_mb5.json"
# bash shell/exp_shell/franka/franka_new_bs32_20ep_baseline_pt_mb_50samples_v5.sh

# # 6. baseline + pt + mb + dep
# export output_dir="work_dirs_ssd_group/franka_new_bs32_40ep_baseline_pt_mb_aug_dep_50samples_v5_norm_crop_unfrozen_embed_40ep_task5_new_drop08_60ep"
# export meta_path="shell/exp_data/franka_task5_50samples_mb5.json"
# bash shell/exp_shell/franka/franka_new_bs32_20ep_baseline_pt_mb_aug_dep_50samples_v5.sh


# # 1. openvla
# cd /SSD_DISK/users/zhangjiahui/openvla
# export data_root_dit="/SSD_DISK/users/zhangjiahui/openvla/data/franka/openvla_data_task_mv"
# export max_steps=1000
# export save_steps=1000
# bash vla-scripts/finetune.sh


# # cd /SSD_DISK/users/zhangjiahui/openvla/internvl_chat
# # 2. baseline
# export output_dit="work_dirs_ssd_group/franka_new_bs32_20ep_baseline_50samples_v5_norm_repeatimg_task_mv"
# export meta_path="shell/exp_data/franka_task_mv_50samples_hist1.json"
# bash shell/exp_shell/franka/franka_new_bs32_20ep_baseline_50samples_v5.sh

# # 3. baseline + pt
# export output_dit="work_dirs_ssd_group/franka_new_bs32_20ep_baseline_pt_50samples_v5_norm_task_mv"
# export meta_path="shell/exp_data/franka_task_mv_50samples_hist1.json"
# bash shell/exp_shell/franka/franka_new_bs32_20ep_baseline_pt_50samples_v5.sh

# # # 4. baseline + pt + dep
# export output_dit="work_dirs_ssd_group/franka_new_bs32_20ep_baseline_pt_dep_50samples_v5_norm_no_crop_task_mv_aug"
# export meta_path="shell/exp_data/franka_task_mv_50samples_hist1.json"
# bash shell/exp_shell/franka/franka_new_bs32_20ep_baseline_pt_dep_50samples_v5.sh

# # 5. baseline + pt + mb
# export output_dit="work_dirs_ssd_group/franka_new_bs32_20ep_baseline_pt_mb_50samples_v5_norm_task_mv_aug_drop08_30ep"
# export meta_path="shell/exp_data/franka_task_mv_50samples_mb5.json"
# bash shell/exp_shell/franka/franka_new_bs32_20ep_baseline_pt_mb_50samples_v5.sh

# 6. baseline + pt + mb + dep
export output_dir="work_dirs_ssd_group/franka_new_bs32_40ep_baseline_pt_mb_aug_dep_50samples_v5_norm_crop_unfrozen_embed_40ep_task_mv_new_drop08_60ep_crop"
export meta_path="shell/exp_data/franka_task_mv_50samples_mb5.json"
bash shell/exp_shell/franka/franka_new_bs32_20ep_baseline_pt_mb_aug_dep_50samples_v5.sh