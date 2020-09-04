import os
from glob import glob


xml = r"""
<annotation verified="yes">
	<folder>raw</folder>
	<filename>{filename}.jpg</filename>
	<path>C:\Users\Techfast\PycharmProjects\tft_object_detection\data\vid_1\raw\{filename}.jpg</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>1280</width>
		<height>720</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>player</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>1176</xmin>
			<ymin>168</ymin>
			<xmax>1277</xmax>
			<ymax>229</ymax>
		</bndbox>
	</object>
	<object>
		<name>trait</name>
		<pose>Unspecified</pose>
		<truncated>1</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>1</xmin>
			<ymin>186</ymin>
			<xmax>121</xmax>
			<ymax>222</ymax>
		</bndbox>
	</object>
	<object>
		<name>trait</name>
		<pose>Unspecified</pose>
		<truncated>1</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>1</xmin>
			<ymin>222</ymin>
			<xmax>136</xmax>
			<ymax>259</ymax>
		</bndbox>
	</object>
	<object>
		<name>trait</name>
		<pose>Unspecified</pose>
		<truncated>1</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>1</xmin>
			<ymin>256</ymin>
			<xmax>121</xmax>
			<ymax>292</ymax>
		</bndbox>
	</object>
	<object>
		<name>trait</name>
		<pose>Unspecified</pose>
		<truncated>1</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>1</xmin>
			<ymin>292</ymin>
			<xmax>137</xmax>
			<ymax>331</ymax>
		</bndbox>
	</object>
</annotation>
"""

if __name__ == "__main__":
    files = glob('./data/vid_1/raw/*.jpg')
    for file_name in files:
        file_path, jpgname = os.path.split(file_name)
        xml_i = xml.format(filename=jpgname)
        xml_name = file_name.replace(r'.jpg', r'.xml')
        with open(xml_name, 'w') as f:
            print(jpgname)
            f.write(xml_i)