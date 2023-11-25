from typing import Optional

from tracker.tracker_zoo import load_tracker



def load_tracker_module(config_path, tracker_type, tracker_config_path, tracker_weight_path: Optional[str] = None):
    if tracker_type == "STRONGSORT":
        tracker_module = load_tracker(
            config_path=config_path,
            tracker_type=tracker_type,
            tracker_weight_path=tracker_weight_path,
            tracker_config_path=tracker_config_path,
        )

    else:
        tracker_module = load_tracker(
            config_path=config_path,
            tracker_type=tracker_type,
            tracker_config_path=tracker_config_path,
        )

    return tracker_module
