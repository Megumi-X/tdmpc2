from ast import List
from typing import Optional
import numpy as np
import xml.etree.ElementTree as ET

class TerrainGenerator():
    def __init__(self, xml_string: str):
        self.blank_xml_string = xml_string

    def reset_terrains(self,
                       ceils_start_pos: list,
                       ceils_end_pos: list,
                       mounts_start_pos: list,
                       mounts_end_pos: list,
                       mounts_heights: list) -> str:
        root = ET.fromstring(self.blank_xml_string)
        worldbody = root.find('worldbody')

        # Sanity checks
        if worldbody is None:
            raise ValueError("No <worldbody> element found in the XML.")
        if len(ceils_start_pos) != len(ceils_end_pos):
            raise ValueError("Length of ceils_start_pos and ceils_end_pos must be the same.")
        if len(mounts_start_pos) != len(mounts_end_pos) or len(mounts_start_pos) != len(mounts_heights):
            raise ValueError("Length of mounts_start_pos, mounts_end_pos, and mounts_heights must be the same.")
        
        # Add new ceiling
        for i, (start_pos, end_pos) in enumerate(zip(ceils_start_pos, ceils_end_pos)):
            ceil_name = f'ceil_{i}'
            ceil_size = [ (end_pos - start_pos) / 2, 400, 98.6 ]
            ceil_pos = [ (start_pos + end_pos) / 2, 0, 100 ]
            ceil_rgba = [0.5, 0.9, 0.7, 1.0]

            attributes = {
                'name': ceil_name,
                'type': 'box',
                'size': ' '.join(map(str, ceil_size)),
                'pos': ' '.join(map(str, ceil_pos)),
                'rgba': ' '.join(map(str, ceil_rgba)),
                'conaffinity': '1'
            }

            new_ceil_element = ET.Element('geom', attrib=attributes)
            worldbody.append(new_ceil_element)

        # Add new mounts and walls
        for i, (start_pos, end_pos, height) in enumerate(zip(mounts_start_pos, mounts_end_pos, mounts_heights)):
            mount_name = f'mount_{i}'
            mount_size = [ (end_pos - start_pos) / 2, 400, height / 2 ]
            mount_pos = [ (start_pos + end_pos) / 2, 0, height / 2 ]
            mount_rgba = [0.9, 0.7, 0.5, 1.0]

            wall_name = f'wall_{i}'
            wall_size = [ 0.01, 400, height / 2 ]
            wall_pos = [ start_pos, 0, height / 2 ]
            wall_rgba = [1.0, 0.0, 0.0, 0.0]

            mount_attributes = {
                'name': mount_name,
                'type': 'box',
                'size': ' '.join(map(str, mount_size)),
                'pos': ' '.join(map(str, mount_pos)),
                'rgba': ' '.join(map(str, mount_rgba)),
                'conaffinity': '1'
            }

            wall_attributes = {
                'name': wall_name,
                'type': 'box',
                'size': ' '.join(map(str, wall_size)),
                'pos': ' '.join(map(str, wall_pos)),
                'rgba': ' '.join(map(str, wall_rgba)),
                'conaffinity': '1'
            }

            new_mount_element = ET.Element('geom', attrib=mount_attributes)
            worldbody.append(new_mount_element)

            new_wall_element = ET.Element('geom', attrib=wall_attributes)
            worldbody.append(new_wall_element)     
        
        modified_xml_string = ET.tostring(root, encoding='unicode')
        return modified_xml_string
    
    def get_max_min_height_array(self,
                                 ceils_start_pos: list,
                                 ceils_end_pos: list,
                                 mounts_start_pos: list,
                                 mounts_end_pos: list,
                                 mounts_heights: list):
        resolution_start = -300
        resolution_end = 300  # exclusive
        num_bins = resolution_end - resolution_start

        if len(ceils_start_pos) != len(ceils_end_pos):
            raise ValueError("Length of ceils_start_pos and ceils_end_pos must be the same.")
        if len(mounts_start_pos) != len(mounts_end_pos) or len(mounts_start_pos) != len(mounts_heights):
            raise ValueError("Length of mounts_start_pos, mounts_end_pos, and mounts_heights must be the same.")

        if mounts_heights:
            highest_mount = max(mounts_heights)
            default_max_height = highest_mount + 20
        else:
            highest_mount = 30
            default_max_height = 50

        min_heights = np.full(num_bins, 0.125, dtype=float)
        max_heights = np.full(num_bins, default_max_height, dtype=float)

        def apply_segment(start: float, end: float, min_val: float, max_val: float):
            seg_start = max(int(np.floor(start)), resolution_start)
            seg_end = min(int(np.ceil(end)), resolution_end)
            for pos in range(seg_start, seg_end):
                idx = pos - resolution_start
                if 0 <= idx < num_bins:
                    min_heights[idx] = min_val
                    max_heights[idx] = max_val

        for start, end in zip(ceils_start_pos, ceils_end_pos):
            apply_segment(start, end, 0.125, 1.4)
           
        for start, end, height in zip(mounts_start_pos, mounts_end_pos, mounts_heights):
            apply_segment(start, end, height, height + 20)

        return max_heights, min_heights
    
class StairsGenerator():
    def __init__(self, xml_string: str):
        self.blank_xml_string = xml_string

    def reset_terrains(self,
                       hfield: dict,
                       contype: int = 1,
                       conaffinity: int = 1,
                       y_width: float = 400,
                       slope_info: Optional[List] = None,
                       tunnel_info: Optional[List] = None,
                       render_info: Optional[dict] = None,
                       collision_info: Optional[List] = None) -> str:
        root = ET.fromstring(self.blank_xml_string)
        worldbody = root.find('worldbody')

        # Sanity checks
        if worldbody is None:
            raise ValueError("No <worldbody> element found in the XML.")
        
        if render_info is not None:
            visual = root.find('visual')
            if visual is None:
                visual = ET.SubElement(root, 'visual')
            global_tag = visual.find('global')
            if global_tag is None:
                global_tag = ET.SubElement(visual, 'global')    
            global_tag.set('offwidth', f'{render_info["offwidth"]}')
            global_tag.set('offheight', f'{render_info["offheight"]}')
            quality = visual.find('quality')
            if quality is None:
                quality = ET.SubElement(visual, 'quality')
            quality.set('shadowsize', f'{render_info["shadowsize"]}')
        
        i = 0
        # Add new stairs
        for key, value in hfield.items():
            start_pos = key[0]
            end_pos = key[1]
            height = value
            stair_name = f'stair_{i}'
            stair_size = [ (end_pos - start_pos) / 2, y_width, (height + 0.2) / 2 ]
            if stair_size[0] < 1e-6 or stair_size[2] < 1e-6:
                continue
            stair_pos = [ (start_pos + end_pos) / 2, 0, (height + 0.2) / 2 - 0.2 ]
            stair_rgba = [1., 1., 1., 1.]

            attributes = {
                'name': stair_name,
                'type': 'box',
                'size': ' '.join(map(str, stair_size)),
                'pos': ' '.join(map(str, stair_pos)),
                'rgba': ' '.join(map(str, stair_rgba)),
                'conaffinity': f'{conaffinity}',
                'contype': f'{contype}',
                'material': 'grid_gd',
                'friction': '1.0 0.05 0.01'
            }

            new_stair_element = ET.Element('geom', attrib=attributes)
            worldbody.append(new_stair_element)
            i += 1


        if slope_info is not None:
            for j, slope in enumerate(slope_info):
                start_x = slope['start_x']
                end_x = slope['end_x']
                start_z = slope['start_z']
                end_z = slope['end_z']
                height = end_z - start_z
                name = f'slope_{j}'
                rot_angle = np.arctan(height / (end_x - start_x))
                rot_angle_deg = np.degrees(rot_angle)
                size = [height / 2 / np.sin(rot_angle), y_width, np.abs(height) * np.cos(rot_angle) / 2]
                pos = [(start_x + end_x) / 2 + size[2] * np.sin(rot_angle), 0, (start_z + end_z) / 2 - size[2] * np.cos(rot_angle)]
                rgba = [1., 1., 1., 1.]

                attributes = {
                    'name': name,
                    'type': 'box',
                    'size': ' '.join(map(str, size)),
                    'pos': ' '.join(map(str, pos)),
                    'rgba': ' '.join(map(str, rgba)),
                    'conaffinity': f'{conaffinity}',
                    'contype': f'{contype}',
                    'material': 'grid_gd',
                    'friction': '1.0 0.05 0.01',
                    'euler': f'0 {-rot_angle_deg} 0'
                }

                new_slope_element = ET.Element('geom', attrib=attributes)
                worldbody.append(new_slope_element)

        if tunnel_info is not None:
            for j, tunnel in enumerate(tunnel_info):
                start_x = tunnel['start_x']
                end_x = tunnel['end_x']
                height = tunnel['height']
                y_width = tunnel['y_width']
                thick = tunnel['thick']

                name_ceil = f'tunnel_{j}_ceiling'
                pos_ceil = [(start_x + end_x) / 2, 0, height + thick / 2 - 0.2]
                size_ceil = [(end_x - start_x) / 2, y_width, thick / 2]
                name_sides = [f'tunnel_{j}_side_{k}' for k in range(2)]
                pos_sides = [[(start_x + end_x) / 2, (-1)**k * (y_width + thick) / 2, (height + thick) / 2 - 0.2] for k in range(2)]
                size_sides = [(end_x - start_x) / 2, thick / 2, (height + thick) / 2]

                rgba = [0., 1., 1., .7]

                attributes_ceil = {
                    'name': name_ceil,
                    'type': 'box',
                    'size': ' '.join(map(str, size_ceil)),
                    'pos': ' '.join(map(str, pos_ceil)),
                    'rgba': ' '.join(map(str, rgba)),
                    'conaffinity': f'{conaffinity}',
                    'contype': f'{contype}',
                    'friction': '1.0 0.05 0.01',
                }

                attributes_sides = [{
                    'name': name_sides[k],
                    'type': 'box',
                    'size': ' '.join(map(str, size_sides)),
                    'pos': ' '.join(map(str, pos_sides[k])),
                    'rgba': ' '.join(map(str, rgba)),
                    'conaffinity': f'{conaffinity}',
                    'contype': f'{contype}',
                    'friction': '1.0 0.05 0.01',
                } for k in range(2)]

                new_tunnel_element = ET.Element('geom', attrib=attributes_ceil)
                worldbody.append(new_tunnel_element)
                for attr in attributes_sides:
                    new_side_element = ET.Element('geom', attrib=attr)
                    worldbody.append(new_side_element)

        if collision_info is not None:
            for i, coll in enumerate(collision_info):
                start_x = coll['start_x']
                end_x = coll['end_x']
                height = coll['height']
                names = [f'collision_{i}_0', f'collision_{i}_1']
                poses = [[(start_x + end_x) / 2, y_width + 10, height], [(start_x + end_x) / 2, - (y_width + 10), height]]
                size = [(end_x - start_x) / 2, 10, 2e-3]
                rgba = [0., 0., 0., 0.]
                attributes = [{
                    'name': names[j],
                    'type': 'box',
                    'size': ' '.join(map(str, size)),
                    'pos': ' '.join(map(str, poses[j])),
                    'rgba': ' '.join(map(str, rgba)),
                    'conaffinity': f'{conaffinity}',
                    'contype': f'{contype}',
                } for j in range(2)]
                for attr in attributes:
                    new_collision_element = ET.Element('geom', attrib=attr)
                    worldbody.append(new_collision_element)
                

        modified_xml_string = ET.tostring(root, encoding='unicode')
        return modified_xml_string
    
    def get_max_min_height_array(self,
                                 hfield: dict,
                                 slope_info: Optional[List] = None,
                                 tunnel_info: Optional[List] = None,
                                 unit: int = 0.5,):
        resolution_start = int(-300 / unit)
        resolution_end = int(300 / unit)
        num_bins = resolution_end - resolution_start

        default_max_height = 15

        min_heights = np.full(num_bins, -0.2, dtype=float)
        max_heights = np.full(num_bins, default_max_height, dtype=float)

        def apply_segment(start, end, min_val: Optional[float], max_val: Optional[float]):
            seg_start = max(int(np.floor(start / unit)), resolution_start)
            seg_end = min(int(np.ceil(end / unit)), resolution_end)
            for pos in range(seg_start, seg_end):
                idx = pos - resolution_start
                if 0 <= idx < num_bins:
                    if min_val is not None:
                        min_heights[idx] = min_val
                    if max_val is not None:
                        max_heights[idx] = max_val

        for key, value in hfield.items():
            start = key[0]
            end = key[1]
            height = value
            apply_segment(start, end, height, None)

        if slope_info is not None:
            for slope in slope_info:
                start_x = slope['start_x']
                end_x = slope['end_x']
                start_z = slope['start_z']
                end_z = slope['end_z']
                nums = (end_x - start_x) // unit
                rest = (end_x - start_x) % unit
                for i in range(int(nums)):
                    idx = int((start_x + i * unit) / unit) - resolution_start
                    height = start_z + (end_z - start_z) * ((i + 0.5) * unit) / (end_x - start_x)
                    if 0 <= idx < num_bins:
                        min_heights[idx] = height
                if rest > 1e-6:
                    idx = int((start_x + nums * unit) / unit) - resolution_start
                    height = start_z + (end_z - start_z) * ((nums + 0.5) * unit) / (end_x - start_x)
                    if 0 <= idx < num_bins:
                        min_heights[idx] = height

        if tunnel_info is not None:
            for tunnel in tunnel_info:
                start_x = tunnel['start_x']
                end_x = tunnel['end_x']
                height = tunnel['height']
                apply_segment(start_x, end_x, None, height)

        return max_heights, min_heights
        
