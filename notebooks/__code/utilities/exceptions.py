class MetadataError(Exception):
    """Exception raised for errors in the metadata retrieval process."""
    def __init__(self, message="Error retrieving metadata from the file."):
        self.message = message
        super().__init__(self.message)

