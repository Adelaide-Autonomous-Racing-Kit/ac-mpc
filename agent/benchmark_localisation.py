import time

import numpy as np
import tqdm
from loguru import logger

from utils import load
from ace.steering import SteeringGeometry
from localisation.localisation import LocaliseOnTrack
from localisation.visualisation import LocalisationVisualiser
from localisation.tracker import LocalisationTracker

VISUALISE = True

def load_map(file):
    """Loads the generated map"""
    track_dict = np.load(
        file, allow_pickle=True
    ).item()

    tracks = {
        "left": track_dict["outside_track"],
        "right": track_dict["inside_track"],
        "centre": track_dict["centre_track"],
    }
    logger.info(
        f"Loaded map with shapes: {tracks['left'].shape=}"
        + f"{tracks['right'].shape=}, {tracks['centre'].shape=}"
    )
    return tracks

def localisation_error(game_pose, estimated_pose):
    game_pose_in_our_cf = np.array([-game_pose[0], game_pose[2], game_pose[3]])
    estimated_pose = np.array(estimated_pose)

    pos_error = np.sqrt(np.sum((game_pose_in_our_cf[:2] - estimated_pose[:2]) ** 2))

    rot_error = game_pose_in_our_cf[2] - estimated_pose[2]
    rot_error = (rot_error + np.pi) % (2*np.pi) - np.pi

    return pos_error, rot_error

def main():
    cfg = load.yaml("agent/configs/params.yaml")
    tracks = load_map(cfg["mapping"]["map_path"] + ".npy")
    data = np.load("localisation_data/monza_audi_fastlaps.npy", allow_pickle=True).item()
    # data = np.load("localisation_data/monza_audi_threelapthencrash.npy", allow_pickle=True).item()

    vehicle_data = SteeringGeometry(cfg["vehicle"]["data_path"])

    localiser = LocaliseOnTrack(
        vehicle_data,
        tracks["centre"],
        tracks["left"],
        tracks["right"],
        cfg["localisation"],
    )
    tracker = LocalisationTracker(localiser, data)
    visualiser = LocalisationVisualiser(localiser, tracker)

    logger.info("Running localisation benchmark")
    
    for step in tqdm.tqdm(range(len(data))):
        value = data[step]
        control_inputs = value["control_command"]
        track_detections = value["observation"]
        start_time = time.time()
        localiser.step(
            control_command= control_inputs,
            dt=value["dt"],
            observation=track_detections,
        )
        elapsed_time = time.time() - start_time
        tracker.update(elapsed_time)
        track_detections = {"left": track_detections[0], "right": track_detections[1]}
        if VISUALISE:
            visualiser.update(track_detections)
    
    logger.success(f"Percentage of time localised: {tracker.percentage_of_steps_localised_for()}%")
    logger.success(f"Average position error: {tracker.average_position_error():.2f} meters")
    logger.success(f"Average rotation error: {tracker.average_rotation_error() * 180/np.pi:.2f} degrees")
    

    """

    for step in tqdm.tqdm(range(len(data))):
        value = data[step]


        try:
            localiser.step(value["control_command"], value["dt"], value["observation"])
        except:
            break

        if localiser.is_localised():
            estimated_pos, _, _, _ = localiser.estimated_position

            pos_error, rot_error = localisation_error(value["game_pose"][0], estimated_pos)
            position_errors.append(pos_error)
            rotation_errors.append(rot_error)

            logger.info(f"Position error: {pos_error:.2f}, Rotation error: {rot_error * 180/np.pi:.2f}")
            localised_steps += 1
            if first_localised is None:
                first_localised = step

    """

if __name__ == "__main__":
    main()
