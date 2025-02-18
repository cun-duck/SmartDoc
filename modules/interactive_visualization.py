from moviepy.editor import ImageSequenceClip

def create_gif(image_paths, output_path, fps=1):
    clip = ImageSequenceClip(image_paths, fps=fps)
    clip.write_gif(output_path, fps=fps)