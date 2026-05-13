from google.cloud.storage import Client, transfer_manager
# import gdown
from pathlib import Path

DIR_PATH = Path(__file__).parent.resolve()

def _download_bucket_with_transfer_manager(
    bucket_name, prefix=None, destination_directory="", workers=8, max_results=1000
):
    """Download all of the blobs in a bucket, concurrently in a process pool.

    The filename of each blob once downloaded is derived from the blob name and
    the `destination_directory `parameter. For complete control of the filename
    of each blob, use transfer_manager.download_many() instead.

    Directories will be created automatically as needed, for instance to
    accommodate blob names that include slashes.

    https://cloud.google.com/storage/docs/downloading-objects#storage-download-object-python
    """

    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The directory on your computer to which to download all of the files. This
    # string is prepended (with os.path.join()) to the name of each blob to form
    # the full path. Relative paths and absolute paths are both accepted. An
    # empty string means "the current working directory". Note that this
    # parameter allows accepts directory traversal ("../" etc.) and is not
    # intended for unsanitized end user input.
    # destination_directory = ""

    # The maximum number of processes to use for the operation. The performance
    # impact of this value depends on the use case, but smaller files usually
    # benefit from a higher number of processes. Each additional process occupies
    # some CPU and memory resources until finished. Threads can be used instead
    # of processes by passing `worker_type=transfer_manager.THREAD`.
    # workers=8

    # The maximum number of results to fetch from bucket.list_blobs(). This
    # sample code fetches all of the blobs up to max_results and queues them all
    # for download at once. Though they will still be executed in batches up to
    # the processes limit, queueing them all at once can be taxing on system
    # memory if buckets are very large. Adjust max_results as needed for your
    # system environment, or set it to None if you are sure the bucket is not
    # too large to hold in memory easily.
    # max_results=1000


    storage_client = Client.create_anonymous_client()
    bucket = storage_client.bucket(bucket_name)

    blob_names = [blob.name for blob in bucket.list_blobs(max_results=max_results, prefix=prefix)]

    results = transfer_manager.download_many_to_path(
        bucket, blob_names, destination_directory=destination_directory, max_workers=workers
    )

    for name, result in zip(blob_names, results):
        # The results list is either `None` or an exception for each blob in
        # the input list, in order.

        if isinstance(result, Exception):
            print("Failed to download {} due to exception: {}".format(name, result))
        else:
            print("Downloaded {} to {}.".format(name, destination_directory + name))

# def _download_from_drive(url, output, destination_directory):
#     output = f"{destination_directory}/{output}.tgz"
#     gdown.download_folder(url=url, output=output, quiet=False)

def download(dataset, max_results=1000):
    data_path = f"{DIR_PATH}/data/{dataset}"
    Path(data_path).mkdir(exist_ok=True, parents=True)

    if dataset=="language_table":
        _download_bucket_with_transfer_manager("gresearch", prefix="robotics/language_table/0.1.0/", destination_directory=data_path, max_results=max_results)
    else:
        print(f"Download failed. Unknown dataset: {dataset}")

if __name__=="__main__":
    download("language_table", 10)
