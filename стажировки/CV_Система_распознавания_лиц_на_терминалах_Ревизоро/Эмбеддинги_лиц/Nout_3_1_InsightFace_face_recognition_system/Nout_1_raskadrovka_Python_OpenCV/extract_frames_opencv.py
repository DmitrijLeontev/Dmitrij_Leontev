from datetime import timedelta
import cv2
import numpy as np
import os

SAVING_FRAMES_PER_SECOND = 10

def format_timedelta(td):
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return result + ".00".replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return f"{result}.{ms:02}".replace(":", "-")

def get_saving_frames_durations(cap, saving_fps):
    s = []
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s

def main(video_file):
    print(f"Checking if file exists: {video_file}")
    if not os.path.isfile(video_file):
        print(f"Error: File {video_file} does not exist.")
        return
    
    print("File exists, proceeding with video processing.")
    filename, _ = os.path.splitext(video_file)
    filename += "-opencv"
    if not os.path.isdir(filename):
        os.mkdir(filename)
        print(f"Created directory: {filename}")
    else:
        print(f"Directory already exists: {filename}")
    
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_file}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frames per second: {fps}")
    if fps == 0:
        print("Error: FPS value is zero. Check the video file format.")
        return
    
    saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)
    saving_frames_durations = get_saving_frames_durations(cap, saving_frames_per_second)
    print(f"Saving frames at the following durations: {saving_frames_durations}")
    
    count = 0
    while True:
        is_read, frame = cap.read()
        if not is_read:
            print("Finished reading frames.")
            break
        frame_duration = count / fps
        try:
            closest_duration = saving_frames_durations[0]
        except IndexError:
            print("All frame durations have been processed.")
            break
        if frame_duration >= closest_duration:
            frame_duration_formatted = format_timedelta(timedelta(seconds=frame_duration))
            frame_filename = os.path.join(filename, f"frame{frame_duration_formatted}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved frame at {frame_duration_formatted} as {frame_filename}")
            try:
                saving_frames_durations.pop(0)
            except IndexError:
                print("No more frame durations to save.")
        count += 1

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python extract_frames_opencv.py <video_file>")
    else:
        video_file = sys.argv[1]
        print(f"Video file: {video_file}")
        main(video_file)
