/home/zzp/SSD_ping/anaconda3/envs/z1/bin/python /home/zzp/SSD_ping/my-root-path/My-core-python/pytorch-deeplabV3+/cityscapesScripts-master/cityscapesscripts/helpers/labels.py
List of cityscapes labels:
# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!
 
                     name |  id | trainId |       category | categoryId | hasInstances | ignoreInEval|        color
    --------------------------------------------------------------------------------------------------
                unlabeled |   0 |     255 |           void |          0 |            0 |            1 |         (0, 0, 0)         
              ego vehicle |   1 |     255 |           void |          0 |            0 |            1 |         (0, 0, 0)         
     rectification border |   2 |     255 |           void |          0 |            0 |            1 |         (0, 0, 0)         
               out of roi |   3 |     255 |           void |          0 |            0 |            1 |         (0, 0, 0)
                   static |   4 |     255 |           void |          0 |            0 |            1 |         (0, 0, 0)
                  dynamic |   5 |     255 |           void |          0 |            0 |            1 |      (111, 74, 0)
                   ground |   6 |     255 |           void |          0 |            0 |            1 |       (81, 0, 81)
                     road |   7 |       0 |           flat |          1 |            0 |            0 |    (128, 64, 128)
                 sidewalk |   8 |       1 |           flat |          1 |            0 |            0 |    (244, 35, 232)
                  parking |   9 |     255 |           flat |          1 |            0 |            1 |   (250, 170, 160)
               rail track |  10 |     255 |           flat |          1 |            0 |            1 |   (230, 150, 140)
                #  building |  11 |       2 |   construction |          2 |            0 |            0 |      (70, 70, 70)
                    #  wall |  12 |       3 |   construction |          2 |            0 |            0 |   (102, 102, 156)
                    # fence |  13 |       4 |   construction |          2 |            0 |            0 |   (190, 153, 153)
               guard rail |  14 |     255 |   construction |          2 |            0 |            1 |   (180, 165, 180)
                  #  bridge |  15 |     255 |   construction |          2 |            0 |            1 |   (150, 100, 100)
                   tunnel |  16 |     255 |   construction |          2 |            0 |            1 |    (150, 120, 90)
                    #  pole |  17 |       5 |         object |          3 |            0 |            0 |   (153, 153, 153)
                polegroup |  18 |     255 |         object |          3 |            0 |            1 |   (153, 153, 153)
            # traffic light |  19 |       6 |         object |          3 |            0 |            0 |    (250, 170, 30)
            #  traffic sign |  20 |       7 |         object |          3 |            0 |            0 |     (220, 220, 0)
            #    vegetation |  21 |       8 |         nature |          4 |            0 |            0 |    (107, 142, 35)
            #       terrain |  22 |       9 |         nature |          4 |            0 |            0 |   (152, 251, 152)
            #           sky |  23 |      10 |            sky |          5 |            0 |            0 |    (70, 130, 180)
            #        person |  24 |      11 |          human |          6 |            1 |            0 |     (220, 20, 60)
            #         rider |  25 |      12 |          human |          6 |            1 |            0 |       (255, 0, 0)
            #           car |  26 |      13 |        vehicle |          7 |            1 |            0 |       (0, 0, 142)
            #         truck |  27 |      14 |        vehicle |          7 |            1 |            0 |        (0, 0, 70)
            #           bus |  28 |      15 |        vehicle |          7 |            1 |            0 |      (0, 60, 100)
                  caravan |  29 |     255 |        vehicle |          7 |            1 |            1 |        (0, 0, 90)
                  trailer |  30 |     255 |        vehicle |          7 |            1 |            1 |       (0, 0, 110)
              #       train |  31 |      16 |        vehicle |          7 |            1 |            0 |      (0, 80, 100)
              #  motorcycle |  32 |      17 |        vehicle |          7 |            1 |            0 |       (0, 0, 230)
              #     bicycle |  33 |      18 |        vehicle |          7 |            1 |            0 |     (119, 11, 32)
            license plate |  -1 |      -1 |        vehicle |          7 |            0 |            1 |       (0, 0, 142)
 
Example usages:
ID of label 'car': 26
Category of label with ID '26': vehicle
Name of label with trainID '0': road