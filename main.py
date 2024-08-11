from utils import read_video, save_video
from trackers import Tracker

def main():
    # Read Video
    video_frames = read_video('Input_videos/08fd33_4.mp4')

    # Initialize Tracker
    tracker = Tracker('Models/best.pt')

    tracks = tracker.get_object_tracks(video_frames)

    # save_video(video_frames, 'output_videos/output.avi')
    
if __name__ == '__main__':
    main()