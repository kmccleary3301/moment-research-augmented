import os
from pydantic import BaseModel
from typing import List, Optional, NamedTuple
import numpy as np

class FileEntry(BaseModel):
	path: str
	size: int
	extension: str
 
class FileEntryLoaded(BaseModel):
	path: str
	size: int
	extension: str
	load_time: float
	num_series: int
 
def get_file_entry(file_path: str) -> FileEntry:
    assert os.path.exists(file_path), f"File `{file_path}` does not exist"
    file_size = os.path.getsize(file_path)
    file_extension = os.path.splitext(file_path)[1].lower()
    return FileEntry(**{
        'path': file_path,
        'size': file_size,
        'extension': file_extension
    })


def list_files_recursively(directory):
	file_paths : List[FileEntry] = []
	for root, _, files in os.walk(directory):
		for file in files:
			full_path = os.path.join(root, file)
			file_size = os.path.getsize(full_path)
			file_extension = os.path.splitext(full_path)[1].lower()
			file_entry = FileEntry(**{
				'path': full_path,
				'size': file_size,
				'extension': file_extension
			})
            
			file_paths.append(file_entry)

	return file_paths

def format_file_size(size):
	for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
		if size < 1024:
			return f"{size:.1f} {unit}"
		size /= 1024
