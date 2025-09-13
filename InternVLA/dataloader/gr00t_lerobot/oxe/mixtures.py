"""
mixtures.py

Defines a registry of dataset mixtures and weights for the Open-X Embodiment Datasets. Each dataset is associated with
a float "sampling weight"
"""

from typing import Dict, List, Tuple

# fmt: off
OXE_NAMED_MIXTURES: Dict[str, List[Tuple[str, float]]] = {

    "sys1_scaling": [
        ("system1_scaling_14kobj_object_container_fgrandpos_bgrandpos_fgclip_NOrobotbaserandpos_bgcache10_bgclip_3L2obsAlign_add_rewrite_instruction_with_observations_Merge", 1.0),
    ],

    "bench_v6_all_longrange": [
        ("bench_v6_all_longrange_split0_h264", 1.0),
        ("bench_v6_all_longrange_split1_h264", 1.0),
        ("bench_v6_all_longrange_split2_h264", 1.0),
        ("bench_v6_all_longrange_split3_h264", 1.0),
        ("bench_v6_all_longrange_split4_h264", 1.0),
    ],

    # === Bridge V2 Dataset ===
    "bridge": [
        # ("bridge_oxe", 1.0),                                    # Version of Bridge V2 in Open-X GCP Bucket
        ("bridge_orig_1.0.0_lerobot", 1.0),                                   # Original Version of Bridge V2 from Project Website
    ],


    # === [Moderate-Scale] Bridge++ Mixtures ===
    "bridge_rt_1": [
        # ("bridge_oxe", 1.0)                                   # Version of Bridge V2 in Open-X GCP Bucket
        ("bridge_orig_1.0.0_lerobot", 1.0),                                   # Original Version of Bridge V2 from Project Website

        ("fractal20220817_data_0.1.0_lerobot", 1.0),                          # Google RT-1 Robot Data (Large-Scale)
    ],
    
    # === RT-X Mixtures ===
    "rtx": [
        ("fractal20220817_data_0.1.0_lerobot", 0.54087122203),                # Google RT-1 Robot Data (Large-Scale)
        ("kuka_0.1.0_lerobot", 0.8341046294),
        # ("bridge_oxe", 1.0)                                   # Version of Bridge V2 in Open-X GCP Bucket
        ("bridge_orig_1.0.0_lerobot", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("taco_play_0.1.0_lerobot", 2.0),
        ("jaco_play_0.1.0_lerobot", 2.0),
        ("berkeley_cable_routing_0.1.0_lerobot", 3.0),
        ("roboturk_0.1.0_lerobot", 1.0),
        # ("nyu_door_opening_surprising_effectiveness_0.1.0_lerobot", 5.0),   # Note --> only contains wrist camera images (skip?)
        ("viola_0.1.0_lerobot", 2.0),
        ("berkeley_autolab_ur5_0.1.0_lerobot", 1.0),
        ("toto_0.1.0_lerobot", 1.0),
    ],

    "rtx_franka": [
        ("fractal20220817_data_0.1.0_lerobot", 0.54087122203),                # Google RT-1 Robot Data (Large-Scale)
        ("kuka_0.1.0_lerobot", 0.8341046294),
        # ("bridge_oxe", 1.0)                                   # Version of Bridge V2 in Open-X GCP Bucket
        ("bridge_orig_1.0.0_lerobot", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("taco_play_0.1.0_lerobot", 2.0),
        ("jaco_play_0.1.0_lerobot", 2.0),
        ("berkeley_cable_routing_0.1.0_lerobot", 3.0),
        ("roboturk_0.1.0_lerobot", 1.0),
        # ("nyu_door_opening_surprising_effectiveness_0.1.0_lerobot", 5.0),   # Note --> only contains wrist camera images (skip?)
        ("viola_0.1.0_lerobot", 2.0),
        ("berkeley_autolab_ur5_0.1.0_lerobot", 1.0),
        ("toto_0.1.0_lerobot", 1.0),

        ("taco_play_0.1.0_lerobot", 1.0),
        ("berkeley_cable_routing_0.1.0_lerobot", 1.0),
        ("viola_0.1.0_lerobot", 1.0),
        ("toto_0.1.0_lerobot", 1.0),
        ("stanford_hydra_dataset_converted_externally_to_rlds_0.1.0_lerobot", 1.0),
        ("austin_buds_dataset_converted_externally_to_rlds_0.1.0_lerobot", 3.0),
        ("nyu_franka_play_dataset_converted_externally_to_rlds_0.1.0_lerobot", 3.0),
        ("maniskill_dataset_converted_externally_to_rlds", 0.1),
        ("furniture_bench_dataset_converted_externally_to_rlds_0.1.0_lerobot", 0.1),
        ("cmu_franka_exploration_dataset_converted_externally_to_rlds", 5.0),
        ("austin_sailor_dataset_converted_externally_to_rlds_0.1.0_lerobot", 1.0),
        ("austin_sirius_dataset_converted_externally_to_rlds_0.1.0_lerobot", 1.0),
        ("berkeley_rpt_converted_externally_to_rlds_0.1.0_lerobot", 1.0),
        ("kaist_nonprehensile_converted_externally_to_rlds", 3.0),
        ("stanford_robocook_converted_externally_to_rlds", 1.0),
        ("iamlab_cmu_pickup_insert_converted_externally_to_rlds_0.1.0_lerobot", 1.0),
        ("utaustin_mutex_0.1.0_lerobot", 1.0),
        ("cmu_play_fusion_0.1.0_lerobot", 1.0),
    ],

    # === Open-X Magic Soup ===
    "oxe_magic_soup": [
        ("fractal20220817_data_0.1.0_lerobot", 0.54087122203),                # Google RT-1 Robot Data (Large-Scale)
        ("kuka_0.1.0_lerobot", 0.8341046294),
        # ("bridge_oxe", 1.0)                                   # Version of Bridge V2 in Open-X GCP Bucket
        ("bridge_orig_1.0.0_lerobot", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("taco_play_0.1.0_lerobot", 2.0),
        ("jaco_play_0.1.0_lerobot", 1.0),
        ("berkeley_cable_routing_0.1.0_lerobot", 1.0),
        ("roboturk_0.1.0_lerobot", 2.0),
        # ("nyu_door_opening_surprising_effectiveness_0.1.0_lerobot", 1.0),   # Note --> only contains wrist camera images (skip?)
        ("viola_0.1.0_lerobot", 2.0),
        ("berkeley_autolab_ur5_0.1.0_lerobot", 2.0),
        ("toto_0.1.0_lerobot", 1.0),
        ("language_table_0.1.0_lerobot", 0.1),
        ("stanford_hydra_dataset_converted_externally_to_rlds_0.1.0_lerobot", 2.0),
        ("austin_buds_dataset_converted_externally_to_rlds_0.1.0_lerobot", 1.0),
        ("nyu_franka_play_dataset_converted_externally_to_rlds_0.1.0_lerobot", 3.0),
        ("furniture_bench_dataset_converted_externally_to_rlds_0.1.0_lerobot", 0.1),
        ("ucsd_kitchen_dataset_converted_externally_to_rlds_0.1.0_lerobot", 2.0),
        ("austin_sailor_dataset_converted_externally_to_rlds_0.1.0_lerobot", 1.0),
        ("austin_sirius_dataset_converted_externally_to_rlds_0.1.0_lerobot", 1.0),
        # ("bc_z_0.1.0_lerobot", 0.2),                                        # Note --> raw data is broken!
        ("dlr_edan_shared_control_converted_externally_to_rlds_0.1.0_lerobot", 1.0),
        ("iamlab_cmu_pickup_insert_converted_externally_to_rlds_0.1.0_lerobot", 1.0),
        # ("uiuc_d3field", 1.0),                                # Note --> raw data is broken!
        ("utaustin_mutex_0.1.0_lerobot", 1.0),
        ("berkeley_fanuc_manipulation_0.1.0_lerobot", 2.0),
        ("cmu_stretch_0.1.0_lerobot", 1.0),
    ],

    # === Open-X Magic Soup++ ===
    "oxe_magic_soup_plus": [
        ("fractal20220817_data_0.1.0_lerobot", 0.54087122203),                # Google RT-1 Robot Data (Large-Scale)
        ("kuka_0.1.0_lerobot", 0.8341046294),
        ("bridge_orig_1.0.0_lerobot", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("taco_play_0.1.0_lerobot", 2.0),
        ("jaco_play_0.1.0_lerobot", 1.0),
        ("berkeley_cable_routing_0.1.0_lerobot", 1.0),
        ("roboturk_0.1.0_lerobot", 2.0),
        ("viola_0.1.0_lerobot", 2.0),
        ("berkeley_autolab_ur5_0.1.0_lerobot", 2.0),
        ("toto_0.1.0_lerobot", 1.0),
        ("language_table_0.1.0_lerobot", 0.1),
        ("stanford_hydra_dataset_converted_externally_to_rlds_0.1.0_lerobot", 2.0),
        ("austin_buds_dataset_converted_externally_to_rlds_0.1.0_lerobot", 1.0),
        ("nyu_franka_play_dataset_converted_externally_to_rlds_0.1.0_lerobot", 3.0),
        ("furniture_bench_dataset_converted_externally_to_rlds_0.1.0_lerobot", 0.1),
        ("ucsd_kitchen_dataset_converted_externally_to_rlds_0.1.0_lerobot", 2.0),
        ("austin_sailor_dataset_converted_externally_to_rlds_0.1.0_lerobot", 1.0),
        ("austin_sirius_dataset_converted_externally_to_rlds_0.1.0_lerobot", 1.0),
        ("dlr_edan_shared_control_converted_externally_to_rlds_0.1.0_lerobot", 1.0),
        ("iamlab_cmu_pickup_insert_converted_externally_to_rlds_0.1.0_lerobot", 1.0),
        ("utaustin_mutex_0.1.0_lerobot", 1.0),
        ("berkeley_fanuc_manipulation_0.1.0_lerobot", 2.0),
        ("cmu_stretch_0.1.0_lerobot", 1.0),
        ## New Datasets in MagicSoup++
        ("bc_z_0.1.0_lerobot", 0.2),                                          # Note: use v0.1.0 --> later versions broken
        ("fmb_dataset_1.0.0_lerobot", 1.0),
        ("dobbe_0.0.1_lerobot", 0.2),
        ("droid_1.0.0_lerobot", 0.06),
    ],

    "oxe_magic_soup_plus_minus": [
        ("fractal20220817_data_0.1.0_lerobot", 1.0),                          # Google RT-1 Robot Data (Large-Scale)
        ("kuka_0.1.0_lerobot", 0.8341046294),
        ("bridge_orig_1.0.0_lerobot", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("taco_play_0.1.0_lerobot", 2.0),
        ("jaco_play_0.1.0_lerobot", 1.0),
        ("berkeley_cable_routing_0.1.0_lerobot", 1.0),
        ("roboturk_0.1.0_lerobot", 2.0),
        ("viola_0.1.0_lerobot", 2.0),
        ("berkeley_autolab_ur5_0.1.0_lerobot", 2.0),
        ("toto_0.1.0_lerobot", 1.0),
        # ("language_table_0.1.0_lerobot", 0.1),
        ("stanford_hydra_dataset_converted_externally_to_rlds_0.1.0_lerobot", 2.0),
        ("austin_buds_dataset_converted_externally_to_rlds_0.1.0_lerobot", 1.0),
        ("nyu_franka_play_dataset_converted_externally_to_rlds_0.1.0_lerobot", 3.0),
        ("furniture_bench_dataset_converted_externally_to_rlds_0.1.0_lerobot", 0.1),
        ("ucsd_kitchen_dataset_converted_externally_to_rlds_0.1.0_lerobot", 2.0),
        ("austin_sailor_dataset_converted_externally_to_rlds_0.1.0_lerobot", 1.0),
        ("austin_sirius_dataset_converted_externally_to_rlds_0.1.0_lerobot", 1.0),
        ("dlr_edan_shared_control_converted_externally_to_rlds_0.1.0_lerobot", 1.0),
        ("iamlab_cmu_pickup_insert_converted_externally_to_rlds_0.1.0_lerobot", 1.0),
        ("utaustin_mutex_0.1.0_lerobot", 1.0),
        ("berkeley_fanuc_manipulation_0.1.0_lerobot", 2.0),
        ("cmu_stretch_0.1.0_lerobot", 1.0),
        ## New Datasets in MagicSoup++
        ("bc_z_0.1.0_lerobot", 0.2),                                          # Note: use v0.1.0 --> later versions broken
        ("fmb_dataset_1.0.0_lerobot", 1.0),
        ("dobbe_0.0.1_lerobot", 0.2),
        # ("droid_1.0.0_lerobot", 0.06),
    ],

    # === T-DROID Dataset ===
    "tdroid_carrot_in_bowl": [
        ("tdroid_carrot_in_bowl", 1.0),
    ],
    "tdroid_pour_corn_in_pot": [
        ("tdroid_pour_corn_in_pot", 1.0),
    ],
    "tdroid_flip_pot_upright": [
        ("tdroid_flip_pot_upright", 1.0),
    ],
    "tdroid_move_object_onto_plate": [
        ("tdroid_move_object_onto_plate", 1.0),
    ],
    "tdroid_knock_object_over": [
        ("tdroid_knock_object_over", 1.0),
    ],
    "tdroid_cover_object_with_towel": [
        ("tdroid_cover_object_with_towel", 1.0),
    ],

    # === DROID Finetuning Datasets ===
    "droid_wipe": [
        ("droid_wipe", 1.0),
    ],

    # === Custom Finetuning Datasets ===
    "custom_finetuning": [
        ("gen_manip_tiny", 1.0),
    ],
}
# fmt: on
