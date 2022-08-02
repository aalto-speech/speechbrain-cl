import warnings


try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
except ImportError:
    warnings.warn(
        "Could not import matplotlib. If you are planning to use the visualization functions then you need to install it along with seaborn.\n\t pip install matplotlib==3.5.0 seaborn==0.11.2"
    )
    pass
