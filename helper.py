import numpy as np
import xarray as xr


def compute_cell_area(lat, lon, R=6371.0):
    """
    Compute cell areas for a 3D Mercator grid with time-dependent latitude and longitude.
    
    Parameters:
    - lat: xarray DataArray, 3D array of latitudes in degrees with dims (Time, south_north_subgrid, west_east_subgrid)
    - lon: xarray DataArray, 3D array of longitudes in degrees with dims (Time, south_north_subgrid, west_east_subgrid)
    - R: float, Earth's radius in km (default: 6371 km)
    
    Returns:
    - area: xarray DataArray, 3D array of cell areas in km^2 with dims (Time, south_north_subgrid, west_east_subgrid)
    """
    # Convert latitudes and longitudes to radians
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    
    # Compute differences in latitude and longitude
    dlat = lat.diff(dim='south_north_subgrid')  # Differences along meridional direction
    dlon = lon.diff(dim='west_east_subgrid')    # Differences along zonal direction
    
    # Pad the differences to match original array shape
    dlat = xr.concat([dlat, dlat.isel(south_north_subgrid=-1)], dim='south_north_subgrid')
    dlon = xr.concat([dlon, dlon.isel(west_east_subgrid=-1)], dim='west_east_subgrid')
    
    # Convert differences to radians
    dlat_rad = np.deg2rad(dlat)
    dlon_rad = np.deg2rad(dlon)
    
    # Compute the cosine of the latitude at cell centers
    lat_center = lat_rad.rolling(south_north_subgrid=2).mean().shift(south_north_subgrid=-1)
    lat_center = lat_center.fillna(lat_center.isel(south_north_subgrid=-2))
    cos_lat = np.cos(lat_center)
    
    # Compute cell area: R^2 * dlat * dlon * cos(lat_center)
    area = (R ** 2) * dlat_rad * dlon_rad * cos_lat
    
    # Ensure the output is a DataArray with the same coordinates as input
    area = area.rename('cell_area')
    area.attrs['units'] = 'km^2'
    
    return area