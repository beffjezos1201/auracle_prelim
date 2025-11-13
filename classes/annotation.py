import os
from typing import List, Tuple
import pandas as pd
from classes.data import Data  # fallback duration


class Annotation:
    """ Class to store seizure annotations as read in the tsv annotation files from the SeizeIT2 BIDS dataset.
    """
    def __init__(
        self,
        events: List[Tuple[int, int]],
        type: List[str],
        lateralization: List[str],
        localization: List[str],
        vigilance: List[str],
        rec_duration: float
    ):
        """Initiate an annotation instance

        Args:
            events (List([int, int])): list of tuples where each element contains the start and stop times in seconds of the event
            type (List[str]): list of event types according to the dataset's events dictionary (events.json).
            lateralization (List[str]): list of lateralization characteristics of the events according to the dataset's events dictionary (events.json).
            localization (List[str]): list of localization characteristics of the events according to the dataset's events dictionary (events.json).
            vigilance (List[str]): list of vigilance characteristics of the events according to the dataset's events dictionary (events.json).

        Returns:
            Annotation: returns an Annotation instance containing the events of the recording.
        """
        self.events = events
        self.types = type
        self.lateralization = lateralization
        self.localization = localization
        self.vigilance = vigilance
        self.rec_duration = rec_duration

    @classmethod
    def loadAnnotation(
        cls,
        annotation_path: str,
        recording: List[str],
    ):
        szEvents = list()
        szTypes = list()
        szLat = list()
        szLoc = list()
        szVig = list()
        durs = 0.0
        tsvFile = os.path.join(
            annotation_path,
            recording[0],
            'ses-01',
            'eeg',
            '_'.join([recording[0], 'ses-01', 'task-szMonitoring', recording[1], 'events.tsv'])
        )
        if os.path.isfile(tsvFile):
            try:
                df = pd.read_csv(tsvFile, delimiter='\t')
                for _, e in df.iterrows():
                    if e.get('eventType') not in ['bckg', 'impd']:
                        szEvents.append([e['onset'], e['onset'] + e['duration']])
                        szTypes.append(e.get('eventType', ''))
                        szLat.append(e.get('lateralization', ''))
                        szLoc.append(e.get('localization', ''))
                        szVig.append(e.get('vigilance', ''))
                    durs = e.get('recordingDuration', durs)
            except Exception as ex:
                print(f"Annotation parse failed for {recording}: {ex}")
        else:
            # Fallback: estimate duration from mov data if available
            mov_path = os.path.join(annotation_path, recording[0], 'ses-01', 'mov')
            if os.path.isdir(mov_path):
                try:
                    rec_data = Data.loadData(annotation_path, recording, modalities=['mov'])
                    if rec_data and 'mov' in rec_data:
                        # Assume sampling rate 25 Hz unless otherwise known
                        durs = rec_data['mov'].shape[1] / 25.0
                        print(f"Using fallback duration from mov for {recording}: {durs:.1f}s")
                except Exception as ex:
                    print(f"Fallback duration failed for {recording}: {ex}")
            else:
                print(f"Events file missing and mov folder absent for {recording}.")

        return cls(
            szEvents,
            szTypes,
            szLat,
            szLoc,
            szVig,
            durs,
        )
