import ffmpeg
import sys


def convert_mov_to_h264(input_file, output_file):
    try:
        (
            ffmpeg.input(input_file)
            .output(
                output_file, vcodec="libx264", crf=23, preset="medium"
            )  # H.264 codec
            .run(overwrite_output=True)
        )
        print(f"Conversion successful! Output saved as: {output_file}")
    except ffmpeg.Error as e:
        print("Error during conversion:", e.stderr.decode(), file=sys.stderr)
        raise


if __name__ == "__main__":
    input_file = "Videos/4.mov"  # Replace with your .mov file path
    output_file = "Videos/4.mp4"  # Replace with your desired output path
    convert_mov_to_h264(input_file, output_file)
