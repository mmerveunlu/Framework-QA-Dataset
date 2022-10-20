"""
Spoken Wikipedia Dataset

The class is based on HuggingFace Common Voice dataset configuration
https://github.com/huggingface/datasets/tree/master/datasets/common_voice
"""

import os

import datasets
from datasets.tasks import AutomaticSpeechRecognition

# For now, data url is not working,
_DATA_URL = ""
_LOCAL_PATH = "data/{}.tar"
_LOCAL_FILE_NAME = "{}"
_LOCAL_EXT_ARCHIVE = "data/"
_LOCAL_TSV_FILE = "data/{}"

_CITATION = """\

}
"""

_DESCRIPTION = """\
Spoken Wikipedia dataset is an open source dataset, generated with Google TTS model.
"""

_HOMEPAGE = ""

_LICENSE = ""

_LANGUAGES = {
    "example_tr_wiki": {
        "Language": "tr",
        "Date": "2022-10-01",
        "Size": "3.2MB",
        "Version": "1.0.0",
        "Valid_Hr_Total": 0.13,
        "Train_Hr_Total": 0.05
    },
}


class SpokenWikiConfig(datasets.BuilderConfig):
    """BuilderConfig for SpokenWikipedia."""

    def __init__(self, name, sub_version, **kwargs):
        """
        Args:
          data_dir: `string`, the path to the folder containing the files in the
            downloaded .tar
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          **kwargs: keyword arguments forwarded to super.
        """
        self.sub_version = sub_version
        self.language = kwargs.pop("language", None)
        self.date_of_snapshot = kwargs.pop("date", None)
        self.size = kwargs.pop("size", None)
        self.valid_hr_total = kwargs.pop("val_hrs", None)
        self.train_hr_total = kwargs.pop("train_hrs", None)
        self.total_hr_total = kwargs.pop("total_hrs", None)
        description = f"Spoken Wikipedia speech to text dataset in" \
                      f" {self.language} version {self.sub_version}. " \
                      f"The dataset comprises {self.valid_hr_total}  hours of validation " \
                      f"and {self.train_hr_total} hours of training data. " \
                      f"The dataset has a size of {self.size}."
        super(SpokenWikiConfig, self).__init__(
            name=name, version=datasets.Version("1.0.0", ""), description=description, **kwargs
        )


class SpokenWiki(datasets.GeneratorBasedBuilder):
    DEFAULT_WRITER_BATCH_SIZE = 1000
    BUILDER_CONFIGS = [
        SpokenWikiConfig(
            name=lang_id,
            language=_LANGUAGES[lang_id]["Language"],
            sub_version=_LANGUAGES[lang_id]["Version"],
            date=_LANGUAGES[lang_id]["Date"],
            size=_LANGUAGES[lang_id]["Size"],
            val_hrs=_LANGUAGES[lang_id]["Valid_Hr_Total"],
            train_hrs=_LANGUAGES[lang_id]["Train_Hr_Total"]
        )
        for lang_id in _LANGUAGES.keys()
    ]

    def _info(self):
        features = datasets.Features(
            {
                "path": datasets.Value("string"),
                "audio": datasets.Audio(sampling_rate=16000),
                "sentence": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            task_templates=[AutomaticSpeechRecognition(audio_column="audio", transcription_column="sentence")],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # Download the TAR archive that contains the audio files:
        # Merve: For now, not downloading files
        # archive_path = dl_manager.download(_DATA_URL.format(self.config.name))
        archive_path = _LOCAL_PATH.format(self.config.name)

        # First we locate the data using the path within the archive:
        path_to_data = "/".join([_LOCAL_TSV_FILE.format(self.config.name), self.config.language])
        path_to_clips = "/".join([_LOCAL_FILE_NAME.format(self.config.name), self.config.language, "clips"])
        metadata_filepaths = {
            # split: "/".join([path_to_data, f"{split}.tsv"])
            split: "/".join([path_to_data, f"{split}.tsv"])
              for split in ["train", "test"]
        }
        # (Optional) In non-streaming mode, we can extract the archive locally to have actual local audio files:
        # local_extracted_archive = dl_manager.extract(archive_path) if not dl_manager.is_streaming else None
        # Merve:  Since no downloading, no need to extract into local.
        local_extracted_archive = _LOCAL_EXT_ARCHIVE

        # To access the audio data from the TAR archives using the download manager,
        # we have to use the dl_manager.iter_archive method.
        #
        # This is because dl_manager.download_and_extract
        # doesn't work to stream TAR archives in streaming mode.
        # (we have to stream the files of a TAR archive one by one)
        #
        # The iter_archive method returns an iterable of (path_within_archive, file_obj) for every
        # file in the TAR archive.

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "local_extracted_archive": local_extracted_archive,
                    "archive_iterator": dl_manager.iter_archive(
                        archive_path
                    ),  # use iter_archive here to access the files in the TAR archives
                    "metadata_filepath": metadata_filepaths["train"],
                    "path_to_clips": path_to_clips,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "local_extracted_archive": local_extracted_archive,
                    "archive_iterator": dl_manager.iter_archive(
                        archive_path
                    ),  # use iter_archive here to access the files in the TAR archives
                    "metadata_filepath": metadata_filepaths["test"],
                    "path_to_clips": path_to_clips,
                },
            )
        ]

    def _generate_examples(self,
                           local_extracted_archive,
                           archive_iterator,
                           metadata_filepath,
                           path_to_clips):
        """Yields examples."""
        data_fields = list(self._info().features.keys())
        # audio is not a header of the csv files
        data_fields.remove("audio")
        path_idx = data_fields.index("path")

        all_field_values = {}
        metadata_found = False
        # Here we iterate over all the files within the TAR archive:

        # first read tsv files
        with open(metadata_filepath) as f:
            metadata_found = True
            lines = f.readlines()
            headline = lines[0]

            # column_names = headline.strip().split("\t")
            column_names = [x.replace("\n", "").replace("\r", "") for x in headline.split("\t") if x]
            assert (
                    column_names == data_fields
            ), f"The file should have {data_fields} as column names, but has {column_names}"
            for line in lines[1:]:
                # field_values = line.decode("utf-8").strip().split("\t")
                field_values = line.strip().split("\t")
                # set full path for mp3 audio file
                audio_path = "/".join([path_to_clips, field_values[path_idx]])
                # all_field_values[audio_path] = field_values
                all_field_values[audio_path] = field_values[1:]

        for path, f in archive_iterator:
            # Else, read the audio file and yield an example
            if path.startswith(path_to_clips):
                assert metadata_found, "Found audio clips before the metadata CSV file."
                if not all_field_values:
                    break
                if path in all_field_values:
                    # retrieve the metadata corresponding to this audio file
                    field_values = all_field_values[path]

                    # if data is incomplete, fill with empty values
                    # if len(field_values) < len(data_fields):
                    #    field_values += (len(data_fields) - len(field_values)) * ["''"]
                    result = {'path': path, 'sentence': ",".join(field_values)}
                    # result = {key: value for key, value in zip(data_fields, field_values)}

                    # set audio feature
                    path = os.path.join(local_extracted_archive, path) if local_extracted_archive else path
                    result["audio"] = {"path": path, "bytes": f.read()}
                    # set path to None if the audio file doesn't exist locally (i.e. in streaming mode)
                    result["path"] = path if local_extracted_archive else None
                    yield path, result
