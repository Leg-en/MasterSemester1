import arcpy
import shapefile
from arcpy import analysis
from arcpy import management

arcpy.env.workspace = r"C:\workspace\MasterSemester1\WindEnergy\Project\data\study area"
arcpy.env.parallelProcessingFactor = "100%"

processed_data = r"C:\workspace\MasterSemester1\WindEnergy\Project\data\processed_data"

flurstuecke = r"flurstuecke.shp"
potential_areas = r"potential_areas_lpa_400m.shp"
hausumringe = r"hausumringe.shp"

Clipped = processed_data + r"\clipped_data"
analysis.PairwiseClip(flurstuecke, potential_areas, Clipped)

# Erfüllt constraint mit 15m abstand zur flurstücksgrenze
Area_Flurstuecke_bufferd = processed_data + r"\bestehend"
analysis.PairwiseBuffer(Clipped, Area_Flurstuecke_bufferd, "-15 Meters", "NONE", None, "PLANAR", "0 Meters")

# Bufferd_hausumringe = processed_data + r"\bufferd_data"
# analysis.PairwiseBuffer(hausumringe, Bufferd_hausumringe, "50 Meters", "NONE", None, "PLANAR", "0 Meters")
#
# House_Erased_data = processed_data + r"\house_erased"
# analysis.PairwiseErase(Area_Flurstuecke_bufferd, Bufferd_hausumringe, House_Erased_data)


# Ab hier ist die erfüllung des Constraints von mindestens 50m abstand zu einander
bestehend_buffered = processed_data + r"\bestehend_buffered"
analysis.PairwiseBuffer(potential_areas, bestehend_buffered, "50 Meters", "NONE", None, "PLANAR", "0 Meters")

buffered_intersects = processed_data + r"\buffered_intersects"
analysis.PairwiseIntersect(bestehend_buffered, buffered_intersects, "ALL", None, "INPUT")

Intersects_erased = processed_data + r"\Intersects_erased"
analysis.PairwiseErase(Area_Flurstuecke_bufferd, buffered_intersects, Intersects_erased, None)

# Points über das gesamte ausmaß ansetzen und dann entsprechend Clippend. Deutlich Simpler, stärker Limitiert. Daher grobere zellgröße

sf = shapefile.Reader(Intersects_erased)
originCoordinate = "" + str(sf.bbox[0]) + " " + str(sf.bbox[1])
yAxisCoordinate = "" + str(sf.bbox[0]) + " " + str(sf.bbox[1] + 10)
oppositeCoorner = "" + str(sf.bbox[2]) + " " + str(sf.bbox[3])
cell_size = 5
labels = 'LABELS'
# Extent is set by origin and opposite corner - no need to use a template fc
templateExtent = '#'
geometryType = 'POLYLINE'

points = processed_data + r"\points"
management.CreateFishnet(points, originCoordinate, yAxisCoordinate, cell_size, cell_size, None, None, oppositeCoorner,
                         labels, templateExtent, geometryType)

intersection_points = processed_data + r"\inter_points"
analysis.PairwiseClip(points + "_label", Intersects_erased,
                      intersection_points,
                      None)

# Todo: Fundamente mit einbeziehen, Abstände zu gebäuden mit einbeziehen
