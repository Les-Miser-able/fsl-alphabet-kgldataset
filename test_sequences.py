"""Run: python -B -m unittest test_sequences -v. Camera/TensorFlow not needed."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from fsl_sequence import assign_hands, FRAMES, FORMAT, NORMALIZATION, prepare_clip, prepare_still, normalize_sequence
from fsl_categories import load_categories, resolved_vocabulary
from fsl_sequence_training import split_dataset, metrics_report

ROOT = Path(__file__).resolve().parent


def hand():
    x = np.zeros((21, 3), dtype=np.float32)
    x[:, 0] = np.linspace(.3, .5, 21)
    x[:, 1] = np.linspace(.4, .6, 21)
    x[:, 2] = np.linspace(0, -.05, 21)
    return np.stack([x, x + np.array([.25, 0, 0], dtype=np.float32)])


def fixture():
    records, samples, labels = [], [], []
    for signer in range(6):
        for label in ("A", "J", "Z"):
            sample = np.repeat(hand()[None], FRAMES, axis=0)
            sample[:, :, :, 0] += signer * .001
            records.append(dict(label=label, kind="recording", signer=f"s{signer}",
                                sample_hash=f"{signer}-{label}"))
            samples.append(normalize_sequence(sample))
            labels.append(label)
    records.append(dict(label="A", kind="still", signer=None, sample_hash="still-A"))
    samples.append(prepare_still(hand()))
    labels.append("A")
    return np.array(samples), np.array(labels), dict(category="alphabet", vocabulary=resolved_vocabulary(load_categories()["alphabet"], ROOT), vocabulary_config=load_categories()["alphabet"], format=FORMAT, normalization=NORMALIZATION, records=records)


class SequenceTests(unittest.TestCase):
    def test_still_has_no_motion(self):
        x = prepare_still(hand())
        self.assertEqual(x.shape, (30, 42, 4))
        np.testing.assert_allclose(x[0], x[-1])

    def test_translation_trajectory_is_preserved(self):
        raw = np.repeat(hand()[None], 45, axis=0)
        raw[:, :, :, 0] += np.linspace(0, .2, 45)[:, None, None]
        sequence = prepare_clip(raw, np.linspace(0, 1.5, 45))
        self.assertGreater(sequence[-1, 0, 0], sequence[0, 0, 0]+.5)

    def test_whole_sequence_translation_and_scale_invariance(self):
        raw = np.repeat(hand()[None], 30, axis=0)
        raw[:, :, :, 0] += np.linspace(0, .1, 30)[:, None, None]
        np.testing.assert_allclose(normalize_sequence(raw),
                                   normalize_sequence(raw*2 + .1), atol=2e-6)

    def test_short_internal_tracking_gap_is_interpolated(self):
        raw = np.repeat(hand()[None], 30, axis=0)
        raw[12] = np.nan
        self.assertTrue(np.isfinite(prepare_clip(raw, np.linspace(0, 1, 30))).all())

    def test_missing_boundary_is_rejected(self):
        raw = np.repeat(hand()[None], 30, axis=0)
        raw[0] = np.nan
        with self.assertRaisesRegex(ValueError, "boundary"):
            prepare_clip(raw, np.linspace(0, 1, 30))

    def test_excessive_missing_data_is_rejected(self):
        raw = np.repeat(hand()[None], 30, axis=0)
        raw[5:10] = np.nan
        with self.assertRaises(ValueError):
            prepare_clip(raw, np.linspace(0, 1, 30))

    def test_bad_timestamps_and_degenerate_hand_are_rejected(self):
        with self.assertRaises(ValueError):
            prepare_clip(np.repeat(hand()[None], 30, axis=0), np.zeros(30))
        with self.assertRaises(ValueError):
            prepare_still(np.zeros((2, 21, 3)))

    def test_signers_disjoint_stills_train_only(self):
        x, y, manifest = fixture()
        train, val, test = split_dataset(x, y, manifest, allow_partial=True)
        signer_sets = [{manifest["records"][i]["signer"] for i in ids
                        if manifest["records"][i]["kind"]=="recording"} for ids in (train,val,test)]
        self.assertFalse(signer_sets[0] & signer_sets[1])
        self.assertFalse(signer_sets[0] & signer_sets[2])
        self.assertFalse(signer_sets[1] & signer_sets[2])
        self.assertIn(len(y)-1, train)
        self.assertNotIn(len(y)-1, val)
        self.assertNotIn(len(y)-1, test)
        self.assertEqual(len(set(train)|set(val)|set(test)), len(y))

    def test_full_alphabet_is_required_by_default(self):
        with self.assertRaisesRegex(ValueError, "A-Z"):
            split_dataset(*fixture())

    def test_dynamic_stills_rejected(self):
        x, y, manifest = fixture()
        y[-1] = "J"
        manifest["records"][-1]["label"] = "J"
        with self.assertRaisesRegex(ValueError, "J/Z"):
            split_dataset(x, y, manifest, allow_partial=True)

    def test_duplicate_samples_rejected(self):
        x, y, manifest = fixture()
        manifest["records"][1]["sample_hash"] = manifest["records"][0]["sample_hash"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            split_dataset(x, y, manifest, allow_partial=True)

    def test_per_class_contributor_coverage(self):
        x, y, manifest = fixture()
        for r in manifest["records"]:
            if r["label"] == "Z":
                r["signer"] = "only-one"
        with self.assertRaisesRegex(ValueError, "Z needs"):
            split_dataset(x, y, manifest, allow_partial=True)

    def test_metrics(self):
        r = metrics_report(np.array([0,0,1]), np.array([0,1,1]), ["A","J"])
        self.assertAlmostEqual(r["accuracy"], 2/3)
        self.assertEqual(r["confusion_matrix"], [[1,1],[0,1]])

    def test_training_check_only(self):
        x, y, manifest = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            np.save(p/"X.npy", x)
            np.save(p/"y.npy", y)
            (p/"manifest.json").write_text(json.dumps(manifest))
            result = subprocess.run([sys.executable, "-B", str(ROOT/"train-sequences.py"),
                                     "--data", str(p), "--allow-partial", "--check-only"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("validation:", result.stdout)

    def test_prepare_real_clips_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            for i,label in enumerate(("J","Z")):
                folder = p/"clips"/label
                folder.mkdir(parents=True)
                raw = np.repeat(hand()[None], 40, axis=0)
                raw[:, :, :, i] += np.linspace(0, .15, 40)[:, None, None]
                np.savez(folder/"clip.npz", coords=raw, timestamps=np.linspace(0,1.2,40),
                         label=label, signer="s01", category="alphabet", format=FORMAT)
            result = subprocess.run([sys.executable, "-B", str(ROOT/"prepare-sequences.py"),
                                     "--no-images", "--source", str(p/"clips"),
                                     "--output", str(p/"out")], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(np.load(p/"out"/"X.npy").shape, (2,30,42,4))
            self.assertEqual(np.load(p/"out"/"y.npy").tolist(), ["J","Z"])

    def test_single_hand_is_valid_with_explicit_mask(self):
        points = hand()
        points[0] = np.nan
        seq = prepare_still(points)
        np.testing.assert_array_equal(seq[:, :21], 0)
        np.testing.assert_array_equal(seq[:, 21:, 3], 1)
        self.assertTrue(np.isfinite(seq).all())

    def test_shared_reference_preserves_hand_separation(self):
        points = hand()
        seq = prepare_still(points)
        scale = np.linalg.norm(points[0, 9, :2]-points[0, 0, :2])
        self.assertAlmostEqual(float(seq[0,21,0]-seq[0,0,0]), .25/scale, places=5)

    def test_hand_can_enter_and_leave_without_extrapolation(self):
        raw = np.repeat(hand()[None], 30, axis=0)
        raw[:10,1] = np.nan
        raw[20:,1] = np.nan
        seq = prepare_clip(raw, np.linspace(0,1,30))
        np.testing.assert_array_equal(seq[:10,21:], 0)
        np.testing.assert_array_equal(seq[20:,21:], 0)
        np.testing.assert_array_equal(seq[10:20,21:,3], 1)

    def test_long_single_hand_gap_is_masked_not_invented(self):
        raw = np.repeat(hand()[None], 30, axis=0)
        raw[8:22,1] = np.nan
        seq = prepare_clip(raw, np.linspace(0,1,30))
        np.testing.assert_array_equal(seq[10:20,21:], 0)
        np.testing.assert_array_equal(seq[:,:,3][:,:21], 1)

    def test_detection_order_does_not_swap_slots(self):
        from types import SimpleNamespace as S
        left, right = hand()
        points = lambda x: [S(x=float(p[0]),y=float(p[1]),z=float(p[2])) for p in x]
        result = S(hand_landmarks=[points(right),points(left)],
                   handedness=[[S(category_name="Right",score=.99)],[S(category_name="Left",score=.99)]])
        np.testing.assert_allclose(assign_hands(result), hand())

    def test_ambiguous_duplicate_handedness_is_missing(self):
        from types import SimpleNamespace as S
        points = [S(x=float(p[0]),y=float(p[1]),z=float(p[2])) for p in hand()[0]]
        result = S(hand_landmarks=[points,points],
                   handedness=[[S(category_name="Left",score=.99)],[S(category_name="Left",score=.99)]])
        self.assertTrue(np.isnan(assign_hands(result)).all())

    def test_old_one_hand_data_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Re-extract"):
            prepare_clip(np.repeat(hand()[0][None],30,axis=0),np.linspace(0,1,30))
        x,y,manifest=fixture()
        manifest["format"]="fsl-sequence-v1"
        with self.assertRaisesRegex(ValueError, "rebuild"):
            split_dataset(x,y,manifest,allow_partial=True)


if __name__ == "__main__":
    unittest.main()
