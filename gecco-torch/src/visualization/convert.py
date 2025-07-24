#!/usr/bin/env python3
import numpy as np
import argparse
import os
import glob
from pathlib import Path

# Try to import torch, but make it optional
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Only .npy files will be supported.")

def standardize_bbox(pcl, points_per_object):
    pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    np.random.shuffle(pt_indices)
    pcl = pcl[pt_indices] # n by 3
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = ( mins + maxs ) / 2.
    scale = np.amax(maxs-mins)
    print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center)/scale).astype(np.float32) # [-0.5, 0.5]
    return result

xml_head = \
"""
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        
        <sampler type="ldsampler">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="1600"/>
            <integer name="height" value="1200"/>
            <rfilter type="gaussian"/>
            <boolean name="banner" value="false"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>
    
"""

xml_ball_segment = \
"""
    <shape type="sphere">
        <float name="radius" value="0.012"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""

xml_tail = \
"""
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.5"/>
        </transform>
    </shape>
    
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="7,7,7"/>
        </emitter>
    </shape>
</scene>
"""

def colormap(x,y,z):
    vec = np.array([x,y,z])
    vec = np.clip(vec, 0.001,1.0)
    norm = np.sqrt(np.sum(vec**2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]

def generate_xml_scene(pcl, points_per_object=2048):
    """Generate XML scene content from point cloud data."""
    xml_segments = [xml_head]
    
    pcl = standardize_bbox(pcl, points_per_object)
    pcl = pcl[:,[2,0,1]]
    pcl[:,0] *= -1
    pcl[:,2] += 0.0125

    for i in range(pcl.shape[0]):
        color = colormap(pcl[i,0]+0.5,pcl[i,1]+0.5,pcl[i,2]+0.5-0.0125)
        xml_segments.append(xml_ball_segment.format(pcl[i,0],pcl[i,1],pcl[i,2], *color))
    xml_segments.append(xml_tail)

    return str.join('', xml_segments)

def process_point_cloud_file(file_path, output_dir, points_per_object=2048):
    """Process a single point cloud file (.pt or .npy) and generate corresponding .xml file."""
    try:
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pt':
            if not TORCH_AVAILABLE:
                print(f"Skipping {file_path}: PyTorch not available for .pt files")
                return False
            # Load the .pt file
            pcl = torch.load(file_path, map_location='cpu')
            # Convert to numpy if it's a tensor
            if torch.is_tensor(pcl):
                pcl = pcl.numpy()
        elif file_ext == '.npy':
            # Load the .npy file
            pcl = np.load(file_path)
        else:
            print(f"Unsupported file format: {file_ext}")
            return False
        
        # Generate XML content
        xml_content = generate_xml_scene(pcl, points_per_object)
        
        # Create output filename
        filename = Path(file_path).stem
        xml_filename = f"{filename}_scene.xml"
        output_path = os.path.join(output_dir, xml_filename)
        
        # Write XML file
        with open(output_path, 'w') as f:
            f.write(xml_content)
        
        print(f"Processed: {file_path} -> {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert point cloud files (.pt or .npy) to Mitsuba XML scene files')
    parser.add_argument('--input', help='Directory containing point cloud files')
    parser.add_argument('--output', help='Directory to save generated .xml files')
    parser.add_argument('--points', type=int, default=2048, 
                       help='Number of points to sample from each point cloud (default: 2048)')
    parser.add_argument('--pattern', default='*.pt', 
                       help='File pattern to match (default: *.pt, can also use *.npy or *.* for both)')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input):
        print(f"Error: Input directory '{args.input}' does not exist.")
        return 1
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Find all matching files
    pattern = os.path.join(args.input, args.pattern)
    files = glob.glob(pattern)
    
    # If using *.* pattern, filter for supported extensions
    if args.pattern == '*.*':
        files = [f for f in files if Path(f).suffix.lower() in ['.pt', '.npy']]
    
    if not files:
        print(f"No files matching pattern '{args.pattern}' found in '{args.input}'")
        return 1
    
    print(f"Found {len(files)} files to process")
    
    # Process each file
    successful = 0
    failed = 0
    
    for file_path in files:
        if process_point_cloud_file(file_path, args.output, args.points):
            successful += 1
        else:
            failed += 1
    
    print(f"\nProcessing complete:")
    print(f"Successfully processed: {successful} files")
    print(f"Failed: {failed} files")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit(main())
