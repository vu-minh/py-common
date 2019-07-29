""" Simple plot.

Expected usage:

figure = SimplePlot()
figure.title("my title") \
      .plot(lambda: plot.plot(x, y)) 
      .save("foo.png")


But does not work in practice:
+ workflow is to set properties(title), then plot, then save
+ but in the plot function, we do all (and do not have save_name)
"""

from matplotlib import pyplot as plt


class SimplePlot():
    def __init__(self, size=(6, 6)):
        self.title = ""
        self.save_name = ""
        self.size = size

    def title(self, title):
        self.title = title
        return self

    def size(self, size):
        self.size = size
        return self

    def save(self, save_name):
        self.save_name = save_name
        return self

    def _reset(self):
        self.size = (6, 6)
        self.name = ""
        self.title = ""

    def plot(self, plot_func):
        plt.figure(figsize=self.size)

        plot_func()

        plt.title(self.title)

        if self.save_name:
            plt.tight_layout()
            plt.savefig(self.save_name)
        plt.close()

        self._reset()
        return self


# """
# Simple plotting utils with `mlflow` track artifacts.
# """

# from matplotlib import pyplot as plt
# from matplotlib import cm

# import mlflow
# from mlflow import log_param, log_metric, log_artifact

# import os
# from datetime import datetime

# PLOT_DIR = './'


# class SimpleFigure():
#     """
#     Shorthand for plotting simple line with saving and tracking artifact.

#     ```
#         plot_args = {'prefix':'my_new_xp', 'suffix':'.png', 'figsize'=(10,4),
#                      'save_to_file': True, 'track_with_ml_flow': True}
#         # create object handler
#         fig = SimpleFigure(**plot_args)
#         # and plot a simple line
#         fig.name('losses').plot(lambda: plt.plot(my_losses))
#         # then create a new figure with different size to plot other thing
#         fig.size((10,10)).name('scatter').plot(lambda:plt.scatter(X,Y))
#     ```
#     """

#     def __init__(self, prefix='', suffix='.png', figsize=None,
#                  save_to_file=True, track_with_mlflow=True):
#         '''Create SimpleFigure object with name, prefix , suffix and fig size.'''
#         self.figsize = figsize
#         self.fig_name = 'noname00'
#         self.prefix = prefix
#         self.suffix = suffix
#         self.should_save = save_to_file
#         self.should_track = track_with_mlflow

#     def _prepare_dir(self,):
#         now = datetime.now()
#         plot_dir = '{}/{:02d}{:02d}'.format(PLOT_DIR, now.day, now.month)
#         if not os.path.exists(plot_dir):
#             os.mkdir(plot_dir)
#         return plot_dir

#     def size(self, size=None):
#         '''Manually set figure size'''
#         self.figsize = size
#         return self

#     def name(self, fig_name=''):
#         '''Manually set figure name'''
#         self.fig_name = fig_name
#         return self

#     def plot(self, plot_func):
#         '''Ploting function that take a callable `plot_func` and execute it.'''
#         outname = f'{self._prepare_dir()}/{self.prefix}_{self.fig_name}_{self.suffix}'
#         if self.figsize is not None:
#             plt.figure(figsize=self.figsize)

#         plot_func()

#         if self.should_save:
#             plt.tight_layout()
#             plt.savefig(f'{outname}')
#             plt.close()
#         else:
#             plt.show()
#         if self.should_track:
#             mlflow.log_artifact(f'{outname}')
