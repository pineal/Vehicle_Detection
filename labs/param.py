#Define Feature Parameters
color_space = 'YCrCb'
orient = 9 
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial_size = (32, 32)
hist_bins = 32
spatial_feat = True
hist_feat = True
hog_feat = True

overlap = 0.5
ystart = 400
ystop = 656
scale = 1.5
window_size = 96
x_start_stop = [400, None] # Min and max in y to search in slide_window()
y_start_stop = [380, None] # Min and max in y to search in slide_window()

heat_threshold = 1