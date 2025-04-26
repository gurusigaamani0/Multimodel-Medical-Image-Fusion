import numpy as np


def Region_Growing(image, seed_pixel):
    region_points = [[seed_pixel[0], seed_pixel[1]]]
    img_rg = np.zeros((image.shape[0] + 1, image.shape[1] + 1))
    img_rg[seed_pixel[0]][seed_pixel[1]] = 255.0
    # print('\nloop runs till region growing is complete')
    # # print 'starting points',i,j
    count = 0
    x = [-1, 0, 1, -1, 1, -1, 0, 1]
    y = [-1, -1, -1, 0, 0, 1, 1, 1]

    while len(region_points) > 0:

        if count == 0:
            point = region_points.pop(0)
            i = point[0]
            j = point[1]
        # print('\nloop runs till length become zero:')
        # print('len', len(region_points))
        # print 'count',count
        val = image[i][j]
        lt = val - 8
        ht = val + 8
        # print 'value of pixel',val
        for k in range(8):
            # print '\ncomparison val:',val, 'ht',ht,'lt',lt
            if img_rg[i + x[k]][j + y[k]] != 1:
                try:
                    if image[i + x[k]][j + y[k]] > lt and image[i + x[k]][j + y[k]] < ht:
                        # print '\nbelongs to region',arr[i+x[k]][j+y[k]]
                        img_rg[i + x[k]][j + y[k]] = 1
                        p = [0, 0]
                        p[0] = i + x[k]
                        p[1] = j + y[k]
                        if p not in region_points:
                            if 0 < p[0] < image.shape[0] and 0 < p[1] < image.shape[1]:
                                ''' adding points to the region '''
                                region_points.append([i + x[k], j + y[k]])
                    else:
                        # print 'not part of region'
                        img_rg[i + x[k]][j + y[k]] = 0
                except IndexError:
                    continue

        # # print '\npoints list',region_points
        # if region_points:  # if empty_list will evaluate as false.
        #     importer = region_points.pop(0)
        # else:
        # Get next entry? Do something else?
        if region_points:
            point = region_points.pop(0)
            i = point[0]
            j = point[1]
            count = count + 1
    return img_rg