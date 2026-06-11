import logging


class ExportNewConfigurationToHDF5:
    def __init__(self, new_configuration: dict = None,
                 hdf5_file_path: str = "new_configuration.hdf5",
                 source_hdf5_file_path: str = None,
                 new_hdf5_flag: bool = True):
        self.new_configuration = new_configuration
        self.hdf5_file_path = hdf5_file_path
        self.source_hdf5_file_path = source_hdf5_file_path
        self.new_hdf5_flag = new_hdf5_flag

        logging.addLevelName(logging.INFO, "Export New Configuration To HDF5")
        logging.info("Initialized ExportNewConfigurationToHDF5 with parameters:")
        logging.info(
            f"new_configuration keys: {list(self.new_configuration.keys()) if self.new_configuration else 'None'}, "
            f"hdf5_file_path: {self.hdf5_file_path}, "
            f"new_hdf5_flag: {self.new_hdf5_flag}"
        )

    def export(self):
        import copy
        import h5py
        import json
        import numpy as np
        from scipy.ndimage import rotate

        tilt = self.new_configuration.get("tilt", 0.0)
        perform_tilt = self.new_configuration.get("perform_tilt", False)
        apply_tilt = perform_tilt and tilt != 0.0

        # --- Build the updated metadata/config ---
        source_path = self.source_hdf5_file_path if self.new_hdf5_flag else self.hdf5_file_path
        with h5py.File(source_path, "r") as src:
            existing_config_raw = src["metadata/config"][()] if "metadata/config" in src else None
        existing_config = json.loads(existing_config_raw) if existing_config_raw is not None else {}

        updated_config = copy.deepcopy(existing_config)
        for key, value in self.new_configuration.items():
            # the reconstruction-specific config dicts are merged into the
            # existing ones so untouched fields are preserved; everything else
            # is replaced outright
            if key in ("mbirjax_config", "svmbir_config") and isinstance(value, dict):
                if key not in updated_config:
                    updated_config[key] = {}
                updated_config[key].update(value)
            else:
                updated_config[key] = value

        if self.new_hdf5_flag:
            # --- Create a fresh compact HDF5 that mirrors the source exactly ---
            if self.source_hdf5_file_path is None:
                raise ValueError(
                    "source_hdf5_file_path must be provided when new_hdf5_flag=True"
                )
            with h5py.File(self.source_hdf5_file_path, "r") as src, \
                 h5py.File(self.hdf5_file_path, "w") as dst:

                # Copy every top-level group/dataset and their attributes.
                for name in src:
                    src.copy(name, dst)
                for attr_name, attr_val in src.attrs.items():
                    dst.attrs[attr_name] = attr_val

                # Apply tilt correction to the image volume if needed.
                dataset_key = "raw/normalized_images_log"
                if apply_tilt:
                    if dataset_key in dst:
                        data = dst[dataset_key][:]
                        logging.info(
                            f"Applying tilt correction of {tilt}° to "
                            f"{data.shape[0]} projections ..."
                        )
                        corrected = np.array(
                            [
                                rotate(img, angle=tilt, reshape=False, order=1, mode="nearest")
                                for img in data
                            ],
                            dtype=data.dtype,
                        )
                        del dst[dataset_key]
                        dst.create_dataset(dataset_key, data=corrected)
                        logging.info("Tilt correction applied to new HDF5.")
                    else:
                        logging.warning(
                            f"Tilt correction requested but '{dataset_key}' "
                            f"not found in source — skipping."
                        )

                # Verify angles/deg is present.
                if "angles/deg" not in dst:
                    logging.warning("'angles/deg' not found in source HDF5.")
                else:
                    logging.info(f"angles/deg preserved ({dst['angles/deg'].shape[0]} values).")

                # Write updated config.
                if "metadata/config" in dst:
                    del dst["metadata/config"]
                dst.require_group("metadata")
                dst.create_dataset("metadata/config", data=json.dumps(updated_config))

            logging.info(f"New HDF5 created at {self.hdf5_file_path}")

        else:
            # --- Overwrite: modify the existing file in-place ---
            with h5py.File(self.hdf5_file_path, "a") as f:

                dataset_key = "raw/normalized_images_log"
                if apply_tilt:
                    if dataset_key in f:
                        data = f[dataset_key][:]
                        logging.info(
                            f"Applying tilt correction of {tilt}° to "
                            f"{data.shape[0]} projections ..."
                        )
                        corrected = np.array(
                            [
                                rotate(img, angle=tilt, reshape=False, order=1, mode="nearest")
                                for img in data
                            ],
                            dtype=data.dtype,
                        )
                        del f[dataset_key]
                        f.create_dataset(dataset_key, data=corrected)
                        logging.info("Tilt correction applied in-place.")
                    else:
                        logging.warning(
                            f"Tilt correction requested but '{dataset_key}' "
                            f"not found — skipping."
                        )

                # Verify angles/deg is present.
                if "angles/deg" not in f:
                    logging.warning("'angles/deg' not found in HDF5.")
                else:
                    logging.info(f"angles/deg preserved ({f['angles/deg'].shape[0]} values).")

                # Write updated config.
                if "metadata/config" in f:
                    del f["metadata/config"]
                f.require_group("metadata")
                f.create_dataset("metadata/config", data=json.dumps(updated_config))

            logging.info(f"Configuration updated in-place at {self.hdf5_file_path}")