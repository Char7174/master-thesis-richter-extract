import numpy as np
import os
import random
import sys
sys.path.append('../commentaryparser') # location of the module epinet

import tensorflow as tf
import scipy.ndimage as nd

from epinet.recording import Recording
from epinet.recording import TagSelector
from epinet.serialisation import PickleLoader


class RadarDataset:
    """
    Loads the radar recordings from disk and transfers them to a tensorflow dataset.
    From the file to the dataset, various necessary stages are passed.
    The stages of a recording are: file -> recording -> case -> part -> sample -> dataset.
    During each stage, certain transformations are useful.
    The stages with their components (in brackets) and transformations are 
    briefly described below:
        file (complete recording as one file)
            - selection by seizure types
            - selection by patient id and seizure id
            - transfer to recording object
        recording (complete recording as one object with all additional information)
            - get radarframes
            - extract indices for labeling
        case (complete radardata as one array, indices as tuple)
            - determine unusable areas
            - filter median and gaussian
            - replacing outliers
            - standardization
        parts (part, label)
            - cuts case into parts like pre/while/post seizure
            - labeling the parts
            - eliminates parts by variance below threshold or assigns them 
              to a own class
        sample (data, label)
            - slicing the radardata to a specific length
            - restrict the number of samples
            - clipping the frames to focus on relevant data
            - skip frames
            - add one dimension
        dataset (chained generators)
            - use gen() (as a preset chain) or chain your own choice of 
              generators to transfer the files to a tensorflow dataset, 
              the chain must contain the generators that transfer the data to 
              the next stage (this are generators wihtout [])
              chain order of generators, [optional generator], alternatives // :
                [- gen_samples_clipped]*
                 - gen_samples
                 - gen_labeled_parts
                [- gen_cases_standardized]* //
                [- gen_cases_gaussian_filter] // 
                [- gen_cases_median_filter] //
                [- gen_cases_outliers_replaced]*
                 - gen_cases
                [- gen_recordings_plausibility_check]
                 - gen_recordings
                 * module function inside; function can be used for dataset.map() 
                   to do the function later or to replace the function with 
                   other rules, e.g.
                   dataset = dataset.map(lambda x, y: 
                        (replace_outliers(array, whiskers_length_factor=1.5)), y))
    """
    
    def __init__(self,
                 data_folder = 'pickle_patient_data', 
                 file_part_leading_zeros = 2,
                 file_part_seizure = '/Anfall_',
                 folder_part_patient = '/Patient_',
                 median_filter_size = (3, 1, 1),
                 gaussian_sigma = 1.0,
                 num_frames = 120,
                 num_samples = int(3e+6), 
                 offset = (9, 25, 20, 116),
                 pairs = [],
                 patients = [],
                 plus1d = False,
                 rand = True,
                 segment_type_to_class = {'01': 0, '12': 1, '23': 2, '34': 3, 
                                          '13': 1, '14': 1, '24': 2, '04': -1},
                 seizure_types = [''], 
                 skip_frames = 0,
                 unusable_area = 120, # value from kida notebooks
                 variance_threshold = 0.0,
                                    # example values from data analysis:
                                    # 0.0 keeping all data regardless from variance
                                    # 0.0146 when using standardization and replacing outliers
                                    # 7.0e-22 when using median and gaussian filters
                                    # 3.0e-20 when using no filters
                 variance_threshold_class = False,
                 whiskers_length_factor = 100): # value from data analysis
        """
        Args:
            data_folder: Directory in which the source data is located.
            file_part_leading_zeros: leading zeros of the seizure id
            file_part_seizure: part of the file name containing the recording
            folder_part_patient: part of the folder name containing the recording
            gaussian_sigma: sigma for gaussian filter
            median_filter_size: size for median filter
            num_frames: number of frames of each sample - one segment/sequence
            num_samples: total number of samples
            offset: tuple clips a frame to upper, lower, left and right index, 
                default results from data exploration
            pairs: list of pairs of patient id and seizure id, default is all
            patients: the patients ids
            plus1d: if true adds one dimension, e.g., for channel for CONV2DLSTMs
            rand: random arrangement of samples
            segment_type_to_class: segment type goes from index to index, 
                only index equals none are omitted, e.g., type 13 means this 
                segment goes from index 1 to 3, and there is no index 2 in 
                this recording
            seizure_types: list of strings to restrict to named seizure types, 
                default is none
            skip_frames: number of frames to skip
            unusable_area: specifies the area around an index that should not 
                be used
            variance_threshold: eliminates data by variance below threshold or 
                assigns them to a own class
            variance_threshold_class: if true assigning to a own class
            whiskers_length_factor: specifies the whiskers length for replacing
                outliers with procedure using box plot (usually 1.5)
            sample_counter: counts the total of samples
            num_classes: number of classes
            class_distribution: numpy array containing the class distribution
        """

        self.data_folder = data_folder
        self.file_part_leading_zeros = file_part_leading_zeros
        self.file_part_seizure = file_part_seizure
        self.folder_part_patient = folder_part_patient
        self.gaussian_sigma = gaussian_sigma
        self.median_filter_size = median_filter_size
        self.num_frames = num_frames
        self.num_samples = num_samples
        self.offset = offset
        self.pairs = pairs
        self.patients = patients
        self.plus1d = plus1d
        self.rand = rand
        self.segment_type_to_class = segment_type_to_class
        self.seizure_types = seizure_types
        self.skip_frames = skip_frames
        self.unusable_area = unusable_area
        self.variance_threshold = variance_threshold
        self.variance_threshold_class = variance_threshold_class
        self.whiskers_length_factor = whiskers_length_factor

        self.sample_counter = 0
        self.num_classes = self.get_num_classes()
        self.class_distribution = np.zeros(len(self.segment_type_to_class), dtype=int)


    def gen(self):
        """Preset of chained generators. Ready to use."""

        return self.gen_samples_clipped(
               self.gen_samples(
               self.gen_labeled_parts(
               self.gen_cases_standardized(
               self.gen_cases_outliers_replaced(
               self.gen_cases(
               self.gen_recordings_plausibility_check(
               self.gen_recordings())))))))


    def gen_cases(self, recordings):
        """
        Generates cases from recording objects.

        Args:
            recordings: list or generator for recording objects
        Returns:
            radardata: complete recording as an array of frames
            indices: tuple of indices, e.g., start or end of a seizure
        """

        for recording in recordings:
            radardata = recording.frames.frames
            indices = RadarDataset.extract_recording_indices(recording)
        
            yield radardata, indices

# the first 185 lines of code out of 782