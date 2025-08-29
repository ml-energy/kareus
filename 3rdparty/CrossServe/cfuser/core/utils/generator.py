from typing import List, Iterable


def dimension_generator(batch_sizes=(1, 2, 4, 8, 16), resolution_map=None, ratio_map=None):
    # Default resolution map if none provided
    if resolution_map is None:
        resolution_map = {
            "360p": 360,  # SD video
            "720p": 720,  # HD
            "1080p": 1080,  # Full HD
            "1440p": 1440,  # 2K/QHD
            "2160p": 2160,  # 4K UHD
        }

    # Default ratio map if none provided
    if ratio_map is None:
        ratio_map = {
            "16:9": 16 / 9,  # Standard widescreen
            "21:9": 21 / 9,  # Ultra-wide monitors
            "4:3": 4 / 3,  # Legacy/Some tablets
        }

    # Create a list to store all combinations
    combinations = []
    for res, height in resolution_map.items():
        # Round height up to nearest multiple of 16
        height = ((height + 15) // 16) * 16
        for ratio, ratio_value in ratio_map.items():
            # Calculate width and round up to nearest multiple of 16
            width = int(((height * ratio_value + 15) // 16) * 16)
            # Store tuples of (total_pixels, width, height, batch)
            for batch in batch_sizes:
                combinations.append((height * width * batch, width, height, batch))

    combinations.sort(key=lambda x: (x[0], x[3], x[1], x[2]))

    # Yield only the requested dimensions (without total pixels)
    for _, width, height, batch in combinations:
        yield batch, width, height


def any_dimension_generator(list_of_iterables: List[Iterable]):
    """
    Generate all possible combinations from a list of iterables.

    Args:
        list_of_iterables: List of iterables to generate combinations from
                          e.g., [batch_sizes, widths, heights]

    Yields:
        tuple: A combination containing one element from each iterable
    """
    from itertools import product

    # Generate all combinations using itertools.product
    for combination in product(*list_of_iterables):
        yield combination


if __name__ == "__main__":
    # Usage examples:

    # Example 1: Default usage
    for batch, width, height in dimension_generator():
        print(f"Batch: {batch}, Dimensions: {width}x{height}")

    # Example 2: Custom resolutions
    # custom_resolutions = {
    #     "720p": 720,
    #     "1080p": 1080
    # }
    # for batch, width, height in dimension_generator(resolution_map=custom_resolutions):
    #     print(f"Batch: {batch}, Dimensions: {width}x{height}")

    # Example 3: Custom everything
    # custom_batches = (1, 4)
    # custom_resolutions = {"720p": 720}
    # custom_ratios = {"16:9": 16/9}
    # for batch, width, height in dimension_generator(
    #     batch_sizes=custom_batches,
    #     resolution_map=custom_resolutions,
    #     ratio_map=custom_ratios
    # ):
    #     print(f"Batch: {batch}, Dimensions: {width}x{height}")
