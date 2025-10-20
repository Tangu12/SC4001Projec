import time
import os
import csv

class logger():
    def __init__(self):
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.folder = f"Project_results/Test_{timestamp}"
        os.makedirs(self.folder, exist_ok=True)

        self.filenames = {
            "raw": os.path.join(self.folder, f"Raw.txt"),
            "comp": os.path.join(self.folder, f"Compiled.csv"),
            "det" : os.path.join(self.folder, f"Details.csv")
        }
        self.buffers = {
            "raw": [],
            "comp": [],
            "det" : []
        }
    def write(self, message, fileType = "raw"):
        if fileType == "all":
            for key, value in self.filenames.items():
                with open(value, 'a') as f:
                    if isinstance(message, (list, tuple)):
                        for item in message:
                            f.write(str(item) + '\n')
                    else:
                        f.write(str(message) + '\n')
            return

        if fileType == "comp":
            with open(self.filenames[fileType], 'a', newline='') as f:
                writer = csv.writer(f)

                if isinstance(message, (list, tuple)):
                    writer.writerow(message)
                else:
                    writer.writerow([message])
            return

        with open(self.filenames[fileType], 'a') as f:
            if isinstance(message, (list, tuple)):
                for item in message:
                    f.write(str(item) + '\n')
            else:
                f.write(str(message) + '\n')

    def bar(self, fileType = "raw"):
        filename = self.filenames[fileType]
        with open(filename, 'a') as f:
            f.write('-'*30 + '\n')

    def buffer(self, message, fileType="raw"):
        if isinstance(message, list) and fileType in ("comp", "det"):
            self.buffers[fileType].append(message)
        else:
            self.buffers[fileType].append(str(message))

    def write_buffer(self, fileType="raw"):
        filename = self.filenames[fileType]
        data = self.buffers[fileType]

        if not data:
            return

        if fileType == "raw":
            # Write as text lines
            with open(filename, 'a') as f:
                for item in data:
                    f.write(str(item) + '\n')

        elif fileType in ("comp", "det"):
            # Write as CSV rows
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(data)

        # Clear buffer after writing
        self.buffers[fileType] = []

log = logger()
