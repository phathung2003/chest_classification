from tqdm import tqdm

class TqdmComma(tqdm):

    @staticmethod
    def format_meter(n, total, elapsed, **kwargs):

        text = tqdm.format_meter(
            n=n,
            total=total,
            elapsed=elapsed,
            **kwargs
        )

        if total is not None:

            old_value = f"{int(n)}/{int(total)}"
            new_value = f"{int(n):,}/{int(total):,}"
            text = text.replace(old_value, new_value)

        return text