"""Category workflow tests use synthetic data; no webcam or TensorFlow required."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from fsl_categories import load_categories, check_category_metadata
from fsl_sequence import FORMAT, prepare_clip
from test_sequences import hand, fixture
from fsl_sequence_training import split_dataset

ROOT = Path(__file__).resolve().parent


def command(*args):
    result = subprocess.run([sys.executable, "-B", str(ROOT/"fsl.py"), *args],
                            text=True, capture_output=True)
    return result


class CategoryTests(unittest.TestCase):
    def test_starter_categories(self):
        c = load_categories()
        self.assertEqual(set(c), {"alphabet","numbers","family","wh_words"})
        self.assertIn("J", c["alphabet"]["recording_only_classes"])
        self.assertIn("Z", c["alphabet"]["recording_only_classes"])
        self.assertEqual(c["family"]["classes"], [])
        self.assertEqual(c["numbers"]["classes"], [])

    def test_category_mismatch(self):
        c = load_categories()
        with self.assertRaisesRegex(ValueError,"Category mismatch"):
            check_category_metadata({"category":"alphabet","vocabulary":c["alphabet"]},"family",c["family"])

    def test_changed_vocabulary_rejected(self):
        c = load_categories()
        with self.assertRaisesRegex(ValueError,"changed"):
            check_category_metadata({"category":"family","vocabulary":{}},"family",c["family"])

    def test_unknown_category_and_invalid_label_fail_before_camera(self):
        result = command("capture","--category","not_a_category","--label","A","--signer","s01")
        self.assertNotEqual(result.returncode,0)
        self.assertIn("Unknown category", result.stderr)
        result = command("capture","--category","family","--label","../A","--signer","s01")
        self.assertNotEqual(result.returncode,0)
        self.assertIn("not a class",result.stderr)

    def test_config_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"bad.json"
            p.write_text(json.dumps({"schema_version":1,"categories":{"../escape":{"classes":["A","B"]}}}))
            with self.assertRaisesRegex(ValueError,"Invalid category"):
                load_categories(p)

    def test_capture_writes_category_and_label(self):
        spec = importlib.util.spec_from_file_location("capture_test", ROOT/"capture-sequences.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            raw = np.repeat(hand()[None],30,axis=0)
            t = np.linspace(0,1,30)
            def fake_live(callback, camera):
                callback(prepare_clip(raw,t),raw,t)
            with patch.object(module,"live_clips",fake_live), patch.object(sys,"argv",[
                "capture","--category","family","--label","MOTHER","--signer","s01","--output",tmp]):
                module.main()
            saved = list(Path(tmp).glob("MOTHER/*.npz"))
            self.assertEqual(len(saved),1)
            with np.load(saved[0]) as data:
                self.assertEqual(str(data["category"]),"family")
                self.assertEqual(str(data["label"]),"MOTHER")

    def test_family_prepare_train_check_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            labels=["MOTHER","FATHER","BROTHER","SISTER"]
            for signer in range(6):
                for ci,label in enumerate(labels):
                    folder=p/"clips"/label
                    folder.mkdir(parents=True,exist_ok=True)
                    raw=np.repeat(hand()[None],30,axis=0)
                    raw[:,:,8,1] += (ci+1)*.02
                    raw[:,:,12,0] += signer*.005
                    raw[:,:,:,0] += np.linspace(0,.06+.01*signer,30)[:,None,None]
                    np.savez(folder/f"sample_{signer}.npz",coords=raw,timestamps=np.linspace(0,1,30),
                             signer=f"s{signer}",label=label,category="family",format=FORMAT)
            result=command("prepare","--category","family","--no-images","--source",str(p/"clips"),
                           "--output",str(p/"dataset"))
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(np.load(p/"dataset"/"X.npy").shape,(24,30,42,4))
            meta=json.loads((p/"dataset"/"manifest.json").read_text())
            self.assertEqual(meta["category"],"family")
            result=command("train","--category","family","--data",str(p/"dataset"),"--check-only")
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn("validation:",result.stdout)
            result=command("train","--category","alphabet","--data",str(p/"dataset"),"--check-only")
            self.assertNotEqual(result.returncode,0)
            self.assertIn("Category mismatch",result.stderr)

    def test_wrong_category_clip_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            folder=p/"clips"/"MOTHER"
            folder.mkdir(parents=True)
            np.savez(folder/"sample.npz",coords=np.repeat(hand()[None],30,axis=0),
                     timestamps=np.linspace(0,1,30),signer="s01",label="MOTHER",
                     category="wh_words",format=FORMAT)
            result=command("prepare","--category","family","--no-images","--source",str(p/"clips"),
                           "--output",str(p/"dataset"))
            self.assertNotEqual(result.returncode,0)
            self.assertFalse((p/"dataset"/"X.npy").exists())

    def test_recognition_category_checked_before_tensorflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/"metadata.json").write_text(json.dumps({"category":"alphabet","vocabulary":load_categories()["alphabet"]}))
            result=command("recognize","--category","family","--model-dir",str(p))
            self.assertNotEqual(result.returncode,0)
            self.assertIn("Category mismatch",result.stderr)

    def test_unknown_signer_video_is_training_only(self):
        x,y,manifest=fixture()
        extra=dict(label="J",kind="external_video",signer=None,sample_hash="external-J")
        manifest["records"].append(extra)
        x=np.concatenate([x,x[1:2]])
        y=np.append(y,"J")
        train,val,test=split_dataset(x,y,manifest,allow_partial=True)
        self.assertIn(len(y)-1,train)
        self.assertNotIn(len(y)-1,val)
        self.assertNotIn(len(y)-1,test)

    def test_discover_numeric_classes_from_folders(self):
        from fsl_categories import resolved_vocabulary
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/"1").mkdir()
            (p/"2").mkdir()
            spec=resolved_vocabulary(load_categories()["numbers"],p)
            self.assertEqual(spec["classes"],["1","2"])


if __name__=="__main__":
    unittest.main()
