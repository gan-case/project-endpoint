"""
Script to prepare env before exec
"""

test_opts = Namespace()
    
test_opts.exp_dir='output'
test_opts.checkpoint_path='checkpoint/sam_ffhq_aging.pt'
test_opts.data_path='reference'
test_opts.test_batch_size=4
test_opts.test_workers=4
test_opts.target_age='0,10,20,30,40,50,60,70,80'

from multiprocessing import Process
import tarfile
import os
from progress.spinner import MoonSpinner

MODEL_PATHS = {
    "ffhq_aging": {
        "id": "1XyumF6_fdAxFmxpFcmPf-q84LU_22EMC",
        "name": "sam_ffhq_aging.pt",
        "path": "checkpoint"
    },
    "CACD2000": {
        "id": "1JWTqMEiUZ2yNUJJl_5Ctq8SuskVocn51",
        "name": "CACD2000_refined.tar",
        "path": "dataset"
    },
    "AnnoyIndex_Saved_File": {
        "id": "",
        "name": "CACD2000_refined_images_embeddings_clusters.ann",
        "path": "preprocessed_files"
    },
    "dataframe": {
        "id": "1svruXLPaFKEXg45F9wQV7l0cQ3xHCK34",
        "name": "dataframe_names_clusters_only.parquet.gzip",
        "path": "preprocessed_files"
    }
}


def download_file(file_id, file_name, save_path):
    """Function to generate the urls for given params"""
    url = r"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={FILE_ID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={FILE_ID}" -O {SAVE_PATH}/{FILE_NAME} && rm -rf /tmp/cookies.txt""".format(
        FILE_ID=file_id, FILE_NAME=file_name, SAVE_PATH=save_path
    )
    # check_call(['wget', '--load-cookies', '/tmp/cookies.txt', ], stdout=DEVNULL, stderr=subprocess.STDOUT)
    os.system(url)


def download_and_extract_files(file_id, file_name, save_path):
    """Function to generate the urls for given params"""
    url = r"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={FILE_ID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={FILE_ID}" -O {SAVE_PATH}/{FILE_NAME} && rm -rf /tmp/cookies.txt""".format(
        FILE_ID=file_id, FILE_NAME=file_name, SAVE_PATH=save_path
    )
    os.system(url)
    dataset = tarfile.open(save_path + "/" + file_name)
    dataset.extractall(save_path)
    dataset.close()


def download_all_files():
    """
    Function to download all the three files
    """
    programs = []
    for key, details in MODEL_PATHS.items():
        if not os.path.exists(details["path"]):
            os.makedirs(details["path"])
            if (key == "CACD2000" or key == "embeddings"):
                proc = Process(target=download_and_extract_files, args=(
                    details["id"], details["name"], details["path"],))
                programs.append(proc)
                proc.start()
            else:
                proc = Process(target=download_file, args=(
                    details["id"], details["name"], details["path"],))
                programs.append(proc)
                proc.start()

    # with MoonSpinner('Processing…') as bar:
    for proc in programs:
        proc.join()
    # bar.next()

    return "Environent Ready!"


"""
def prepareEnv():

    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    if not os.path.exists("static"):
        os.makedirs("static")

    path = MODEL_PATHS["fittest_dataset"]
    if not os.path.isfile(UPLOAD_FOLDER + "/" + path["name"]):
        downloadFile(path["id"], path["name"], path["path"])

    # download the CACD dataset
    path = MODEL_PATHS["CACD2000"]
    if not os.path.exists(UPLOAD_FOLDER + "/" + 'CACD2000'):
        downloadFile(path["id"], path["name"], path["path"])
        dataset = tarfile.open(path["path"] + "/" + path["name"])
        dataset.extractall(path["path"])
        dataset.close()
"""
