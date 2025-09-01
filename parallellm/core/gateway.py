from parallellm.core.manager import BatchManager
from parallellm.file_io.file_manager import FileManager


class ParallellmGateway:
    def resume_directory(self, directory):
        """
        Resume a BatchManager from a given directory.
        """
        # Logic to resume from the specified directory
        # TODO
        return BatchManager(file_manager=FileManager(directory))


Parallellm = ParallellmGateway()
