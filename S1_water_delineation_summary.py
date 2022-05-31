import pandas as pd
import os
import geopandas as gpd
import glob
import rasterstats
import seaborn as sns
from rasterstats import zonal_stats

print(pd.__version__)
print(gpd.__version__)
print(sns.__version__)
print(rasterstats.__version__)

def summarize_area(maxextent, sa_imagefiles):
    imagename_list = []
    imagedate_list = []
    sum_list = []

    for sa_image in sa_imagefiles:
        sa_imagename = sa_image[-48:-15]
        sa_imagedate = sa_image[-44:-36]
        gdf = gpd.read_file(maxextent)
        stats = zonal_stats(gdf, sa_image, stats='sum')
        sum = (stats[0][
            "sum"]) * 100  # area in sq metres (stats counts pixels , since value of pixels classified as water is 1, otherwise zero)
        sum_list.append(sum)
        imagename_list.append(sa_imagename)
        imagedate_list.append(sa_imagedate)
        print(sum_list)
        print(imagename_list)
        print(imagedate_list)

    return imagedate_list, sum_list, imagename_list
    del gdf
    gc.collect()


wd = '/gws/nopw/j04/jncc_muirburn/users/kf_waterbodies/'

site_dir = os.path.join(wd, 'site_dir')
output_summary = os.path.join(wd, 'output_image/ges')
waterbody_fn_list = ["Loch_na_Crann.shp", "Kinellen_farm.shp", "Dunain.shp", "Bothyhill.shp", "Black_Loch.shp", "Loch_Flemington_trial.shp"]

site_name_list = ["Loch_na_Crann", "Kinellen_farm", "Dunain", "Bothyhill", "Black_Loch", "Loch_Flemington"]

print(output_summary)

imagefiles = glob.glob(os.path.join(output_summary, '*.tif'))
print(imagefiles)

image_ids = []

for image in imagefiles:
    imagename = image[-48:-15]
    image_ids.append(imagename)
print(image_ids)

for count, waterbody_fp in enumerate(waterbody_fn_list):
    data = []
    site_name = site_name_list[count]
    waterbody_fp = os.path.join(site_dir, waterbody_fp)
    site_name_xlsx = site_name + '.xlsx'
    outfile = os.path.join(wd, 'output_summary/', site_name_xlsx)
    print(outfile)
    site_name_pdf = site_name + '.pdf'
    out_fig = os.path.join(wd, 'output_summary/', site_name_pdf)
    a, b, c = summarize_area(waterbody_fp, imagefiles)

    df = pd.DataFrame({'Date': a, 'Area': b, 'Image': c})
    print(df)
    df.insert(0, "Site", site_name)

    df.to_excel(outfile, index=False)
    graph = sns.scatterplot(data=df, x='Date', y='Area')
    graph.set_ylabel("Area (sq metres)")

    fig = graph.get_figure()
    fig.autofmt_xdate(rotation =45)
    fig.tight_layout()
    fig.savefig(out_fig)
    fig.clf()
