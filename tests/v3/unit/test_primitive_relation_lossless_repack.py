from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from numcodecs import Blosc
import numpy as np
import zarr

from mtare_topo.data.primitive_relation_lossless_repack import repack_zarr_group_losslessly


class PrimitiveRelationLosslessRepackTest(unittest.TestCase):
    def test_repack_changes_compressor_but_not_logical_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source_path = root / "source.zarr"; destination_path = root / "destination.zarr"
            source = zarr.open_group(str(source_path), mode="w")
            source.attrs.update({"schema_version": "test_v1", "student_pose_input_forbidden": True})
            old = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
            floating = np.arange(5 * 3 * 4, dtype=np.float32).reshape(5, 3, 4)
            floating.reshape(-1)[7] = np.float32(np.nan)
            integer = np.asarray([0, 1, 65535, 4, 5], dtype=np.uint16)
            source.create_dataset("range_m", data=floating, chunks=(2, 3, 4), compressor=old)
            source.create_dataset("membership", data=integer, chunks=(3,), compressor=old)
            source["range_m"].attrs["units"] = "m"
            new = Blosc(cname="zstd", clevel=9, shuffle=Blosc.SHUFFLE)
            evidence = repack_zarr_group_losslessly(source_path, destination_path, compressor=new)
            destination = zarr.open_group(str(destination_path), mode="r")
            self.assertEqual([value.name for value in evidence], ["membership", "range_m"])
            self.assertEqual(dict(source.attrs), dict(destination.attrs))
            self.assertEqual(dict(source["range_m"].attrs), dict(destination["range_m"].attrs))
            self.assertEqual(source["range_m"].chunks, destination["range_m"].chunks)
            self.assertEqual(destination["range_m"].compressor.clevel, 9)
            np.testing.assert_array_equal(source["range_m"][:].view(np.uint8), destination["range_m"][:].view(np.uint8))
            np.testing.assert_array_equal(source["membership"][:], destination["membership"][:])

    def test_existing_destination_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source_path = root / "source.zarr"; destination_path = root / "destination.zarr"
            source = zarr.open_group(str(source_path), mode="w")
            source.create_dataset("x", data=np.arange(3), chunks=(2,))
            destination_path.mkdir()
            with self.assertRaises(FileExistsError):
                repack_zarr_group_losslessly(
                    source_path, destination_path,
                    compressor=Blosc(cname="zstd", clevel=9, shuffle=Blosc.SHUFFLE),
                )


if __name__ == "__main__":
    unittest.main()
