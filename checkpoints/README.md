# Checkpoints

Large model checkpoints are not stored in Git.

Expected layout:

    checkpoints/refer/refer_miou_best.state_dict.pth
    checkpoints/refer/refer_oiou_best.state_dict.pth
    checkpoints/rsrefsegrs/rsrefsegrs_test_miou_best.state_dict.pth
    checkpoints/rsrefsegrs/rsrefsegrs_test_oiou_best.state_dict.pth

The runtime expects sanitized checkpoints ending in .state_dict.pth. Raw training checkpoints may contain optimizer state or project-local metadata and should be converted before release.
