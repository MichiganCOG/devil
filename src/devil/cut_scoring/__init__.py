from ...clips import extract_shots_with_ffprobe


class CutScorer(object):
    """Scores the likelihood of a shot cut/transition existing in the given video."""

    def score_cut(self, ffmpeg_video_path):
        """Scores the likelihood of a shot cut/transition existing in the given video.

        :param ffmpeg_video_path: Explicit or template-based video path supported by FFmpeg (str)
        :return: float
        """
        cuts = extract_shots_with_ffprobe(ffmpeg_video_path, 0.0)
        scores = [score for _, score in cuts]

        return max(scores)
