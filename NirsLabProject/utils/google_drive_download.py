import pickle
import os
import shutil

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import io
from tqdm import tqdm
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import re

from NirsLabProject.config import consts
from NirsLabProject.config.consts import FORCE_LOAD_EDF
from NirsLabProject.config.subject import Subject

# To connect to googel drive api
# If modifying these scopes, delete the file token.pickle
CLIENT_SECRET_FILE = os.getenv('GOOGLE_DRIVE_JSON_PATH')
API_NAME = "drive"
API_VERSION = "v3"
SCOPES = ["https://www.googleapis.com/auth/drive"]


class GoogleDriveDownloader:
    def __init__(self):
        self.service = self.create_service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

    # Create a Google Drive API service
    def create_service(self, client_secret_file, api_name, api_version, *scopes):
        CLIENT_SECRET_FILE = client_secret_file
        API_SERVICE_NAME = api_name
        API_VERSION = api_version
        SCOPES = [scope for scope in scopes[0]]
        cred = None

        pickle_file = f"token_{API_SERVICE_NAME}_{API_VERSION}.pickle"

        if os.path.exists(pickle_file):
            with open(pickle_file, "rb") as token:
                cred = pickle.load(token)

        if not cred or not cred.valid:
            if cred and cred.expired and cred.refresh_token:
                cred.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
                cred = flow.run_local_server()

            with open(pickle_file, "wb") as token:
                pickle.dump(cred, token)

        try:
            service = build(API_SERVICE_NAME, API_VERSION, credentials=cred)
            print(API_SERVICE_NAME.capitalize(), "service created successfully.\n")
            return service
        except Exception as e:
            print("Unable to connect.")
            print(e)
            return None

    def downloadfiles(self, dowid, dfilespath, folder=None):
        print(f"Downloading file {dfilespath} to {folder}")
        request = self.service.files().get_media(fileId=dowid)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        pbar = tqdm(total=100, ncols=70)
        while done is False:
            status, done = downloader.next_chunk()
            if status:
                pbar.update(int(status.progress() * 100) - pbar.n)
        pbar.close()
        if folder:
            with io.open(folder + "/" + dfilespath, "wb") as f:
                fh.seek(0)
                f.write(fh.read())
        else:
            with io.open(dfilespath, "wb") as f:
                fh.seek(0)
                f.write(fh.read())

    # List files in folder until all files are found
    def listfolders(self, filid):
        while True:
            results = (
                self.service.files()
                .list(
                    pageSize=1000,
                    q="'" + filid + "'" + " in parents",
                    fields="nextPageToken, files(id, name, mimeType)",
                )
                .execute()
            )
            page_token = results.get("nextPageToken", None)
            if page_token is None:
                folder = results.get("files", [])
            break
        return folder

    # Download folders with files
    def downloadfolders(self, folder_ids):
        for folder_id in folder_ids:
            folder = self.service.files().get(fileId=folder_id).execute()
            folder_name = folder.get("name")
            page_token = None
            while True:
                results = (
                    self.service.files()
                    .list(
                        q=f"'{folder_id}' in parents",
                        spaces="drive",
                        fields="nextPageToken, files(id, name, mimeType)",
                    )
                    .execute()
                )
                page_token = results.get("nextPageToken", None)
                if page_token is None:
                    items = results.get("files", [])
                    # send all items in this section
                    if not items:
                        # download files
                        self.downloadfiles(folder_id, folder_name)
                        print(folder_name)
                    else:
                        # download folders
                        print(f"Start downloading folder '{folder_name}'.")
                        for item in items:
                            if item["mimeType"] == "application/vnd.google-apps.folder":
                                if not os.path.isdir(folder_name):
                                    os.mkdir(folder_name)
                                bfolderpath = os.path.join(os.getcwd(), folder_name)
                                if not os.path.isdir(
                                        os.path.join(bfolderpath, item["name"])
                                ):
                                    os.mkdir(os.path.join(bfolderpath, item["name"]))

                                folderpath = os.path.join(bfolderpath, item["name"])
                                self.listfolders(item["id"], folderpath)
                            else:
                                if not os.path.isdir(folder_name):
                                    os.mkdir(folder_name)
                                bfolderpath = os.path.join(os.getcwd(), folder_name)

                                filepath = os.path.join(bfolderpath, item["name"])
                                self.downloadfiles(item["id"], filepath)
                                print(item["name"])
                break

    def extract_drive_id(self, link):
        # Remove any leading or trailing spaces from the link
        output = []
        link = link.strip()

        if "folders" in link:
            pattern = r"(?<=folders\/)[^|^?]+"
        else:
            pattern = r"(?<=/d/|id=)[^/|^?]+"

        match = re.search(pattern, link)
        return match.group(0)

    def delete_dir_contents(self, dir_path):
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

    def upload_file(self, file_path: str, parent_id: str, name: str = None):
        print(f'Uploading file: {file_path} parent_id: {parent_id}')
        file_metadata = {'name': name or os.path.basename(file_path), 'parents': [parent_id]}
        media = MediaFileUpload(file_path, resumable=True)
        file = self.service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        print(f'File ID: {file.get("id")}')

    def download_subject_data_one_by_one(self, google_drive_link: str, subjects: list = None,  bipolar: bool = False):
        folder_id = self.extract_drive_id(google_drive_link)
        files = self.listfolders(folder_id)

        #  Create a dictionary of subject name to files
        subject_to_files = {}
        for file in files:
            if not file['name'].startswith('p'):
                continue
            subject_name = file['name'].split('_')[0].lower()
            if subjects and subject_name not in subjects:
                continue
            if subject_name not in subject_to_files:
                subject_to_files[subject_name] = []
            #     one layer deeper
            if file['mimeType'] == 'application/vnd.google-apps.folder':
                subject_to_files[subject_name].extend(self.listfolders(file['id']))
            else:
                subject_to_files[subject_name].append(file)

        for subject_name, files in subject_to_files.items():
            subject = Subject(subject_name, bipolar)
            if not consts.DOWNLOAD_FROM_GOOGLE_DRIVE:
                yield subject
                continue

            print(f'Downloading data for subject {subject_name}')
            have_preprocessed_data = False

            for file in files:
                if file['name'].endswith('_pre_processed.zip'):
                    have_preprocessed_data = True
                    break

            for file in files:
                if file['name'].endswith('_pre_processed.zip') and not FORCE_LOAD_EDF:
                    self.downloadfiles(file['id'], file['name'], subject.paths.subject_resampled_data_dir_path)
                    shutil.unpack_archive(os.path.join(subject.paths.subject_resampled_data_dir_path, file['name']), subject.paths.subject_resampled_data_dir_path)
                    os.remove(os.path.join(subject.paths.subject_resampled_data_dir_path, file['name']))
                elif file['name'].endswith('.edf') and (FORCE_LOAD_EDF or not have_preprocessed_data):
                    renamed_file_path = os.path.join(subject.paths.raw_data_dir_path, subject_name.split('_')[0].lower()) + '.edf'
                    if os.path.exists(renamed_file_path):
                        print(f'File {renamed_file_path} already exists, skipping download')
                    elif not os.path.exists(os.path.join(subject.paths.raw_data_dir_path, file['name'])):
                        self.downloadfiles(file['id'], file['name'], subject.paths.raw_data_dir_path)
                    if os.path.exists(os.path.join(subject.paths.raw_data_dir_path, file['name'])):
                        os.rename(
                            os.path.join(subject.paths.raw_data_dir_path,  file['name']),
                            renamed_file_path
                        )
                elif file['name'].endswith('.m'):
                    self.downloadfiles(file['id'], file['name'], subject.paths.hypnogram_data_dir_path)
                elif file['name'].endswith('stim_timing.csv'):
                    self.downloadfiles(file['id'], file['name'], subject.paths.stimuli_dir_path)

            yield subject

            # if not have_preprocessed_data or FORCE_LOAD_EDF:
                # zip the folder
                # zip_name = f'{subject_name}_pre_processed'
                # zip_name_with_suffix = f'{zip_name}.zip'
                # shutil.make_archive(zip_name, 'zip', subject.paths.subject_resampled_data_dir_path)
                # self.upload_file(zip_name_with_suffix, folder_id, zip_name_with_suffix)
                # os.remove(zip_name_with_suffix)
            try:
                os.remove(subject.paths.subject_raw_edf_path)
            except OSError:
                pass
            self.delete_dir_contents(subject.paths.subject_resampled_data_dir_path)
            self.delete_dir_contents(subject.paths.subject_intracranial_model_features_dir_path)

