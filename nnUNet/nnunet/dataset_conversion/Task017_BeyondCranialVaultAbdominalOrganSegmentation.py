#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from collections import OrderedDict
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Pool
import numpy as np
from nnunet.configuration import default_num_threads
from scipy.ndimage import label
from sklearn.model_selection import KFold


def export_segmentations(indir, outdir):
    niftis = subfiles(indir, suffix='nii.gz', join=False)
    for n in niftis:
        identifier = str(n.split("_")[-1][:-7])
        outfname = join(outdir, "test-segmentation-%s.nii" % identifier)
        img = sitk.ReadImage(join(indir, n))
        sitk.WriteImage(img, outfname)


def export_segmentations_postprocess(indir, outdir):
    maybe_mkdir_p(outdir)
    niftis = subfiles(indir, suffix='nii.gz', join=False)
    for n in niftis:
        print("\n", n)
        identifier = str(n.split("_")[-1][:-7])
        outfname = join(outdir, "test-segmentation-%s.nii" % identifier)
        img = sitk.ReadImage(join(indir, n))
        img_npy = sitk.GetArrayFromImage(img)
        lmap, num_objects = label((img_npy > 0).astype(int))
        sizes = []
        for o in range(1, num_objects + 1):
            sizes.append((lmap == o).sum())
        mx = np.argmax(sizes) + 1
        print(sizes)
        img_npy[lmap != mx] = 0
        img_new = sitk.GetImageFromArray(img_npy)
        img_new.CopyInformation(img)
        sitk.WriteImage(img_new, outfname)


if __name__ == "__main__":
    train_dir = "/home/xuzhixiang/xzx/BCV/CoTr_R+A/CoTr-main/data/RawData/Training"
    test_dir = "/home/xuzhixiang/xzx/BCV/CoTr_R+A/CoTr-main/data/RawData/Testing"


    output_folder = "/home/xuzhixiang/xzx/BCV/CoTr_R+A/CoTr-main/data/nnUNet_raw_data_base/nnUNet_raw_data/Task017_BCV"
    img_dir = join(output_folder, "imagesTr")
    lab_dir = join(output_folder, "labelsTr")
    img_dir_te = join(output_folder, "imagesTs")
    maybe_mkdir_p(img_dir)
    maybe_mkdir_p(lab_dir)
    maybe_mkdir_p(img_dir_te)


    def load_save_train(args):
        data_file, seg_file = args
        pat_id = data_file.split("/")[-1]
        pat_id = "bcv_" + str(int(pat_id.split("-")[-1][3:-7]))

        img_itk = sitk.ReadImage(data_file)
        sitk.WriteImage(img_itk, join(img_dir, pat_id + "_0000.nii.gz"))

        img_itk = sitk.ReadImage(seg_file)
        sitk.WriteImage(img_itk, join(lab_dir, pat_id + ".nii.gz"))
        return pat_id

    def load_save_test(args):
        data_file = args
        pat_id = data_file.split("/")[-1]
        pat_id = "bcv_" + str(int(pat_id.split("-")[-1][3:-7]))

        img_itk = sitk.ReadImage(data_file)
        sitk.WriteImage(img_itk, join(img_dir_te, pat_id + "_0000.nii.gz"))
        return pat_id

    nii_files_tr_data = subfiles(join(train_dir,'img'), True, "img", "nii.gz", True)
    nii_files_tr_seg = subfiles(join(train_dir,'label'), True, "label", "nii.gz", True)

    nii_files_ts = subfiles(join(test_dir, 'img'), True, "img", "nii.gz", True)

    p = Pool(default_num_threads)
    train_ids = p.map(load_save_train, zip(nii_files_tr_data, nii_files_tr_seg))
    test_ids = p.map(load_save_test, nii_files_ts)
    p.close()
    p.join()

    json_dict = OrderedDict()
    json_dict['name'] = "BCV"
    json_dict['description'] = "BeyondCranialVaultAbodominalOrganSegmentation"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see challenge website"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT"
    }

    json_dict['labels'] = {
        "0": "background",
        "1": "spleen",
        "2": "kidney_right",
        "3": "kidney_left",
        "4": "gallbladder",
        "5": "esophagus",
        "6": "liver",
        "7": "stomach",
        "8": "aorta",
        "9": "ivc",
        "10": "pvsv",
        "11": "pancreas",
        "12": "adrenal_gland_right",
        "13": "adrenal_gland_left"
    }

    json_dict['numTraining'] = len(train_ids)
    json_dict['numTest'] = len(test_ids)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in train_ids]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_ids]

    with open(os.path.join(output_folder, "dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)

    # create a dummy split (patients need to be separated)
    splits = []
    patients = np.unique([i[:10] for i in train_ids])
    patientids = [i[4:] for i in train_ids]
    splits.append(OrderedDict())
    splits[-1]['train'] = [i for i in train_ids if int(i[4:]) <32]
    splits[-1]['val'] = [i for i in train_ids if int(i[4:]) >= 32]

    save_pickle(splits, join(output_folder, "splits_final.pkl"))