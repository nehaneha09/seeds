# Ultralytics YOLO 🚀, AGPL-3.0 license

from itertools import cycle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from sympy.codegen import Print
from tensorflow.python.ops.signal.shape_ops import frame

from ultralytics.solutions.solutions import BaseSolution  # Import a parent class


class Analytics(BaseSolution):
    """A class to create and update various types of charts (line, bar, pie, area) for visual analytics."""

    def __init__(self, **kwargs):
        """Initialize the Analytics class with various chart types."""
        super().__init__(**kwargs)

        self.bg_color = "#00F344"     # background color of frame
        self.fg_color = "#111E68"     # foreground color of frame
        self.title = "Ultralytics Solutions"
        self.max_points = 45
        self.x_label = "Frame#"
        self.y_label = "Total Counts"
        self.fontsize = 25
        self.type = self.CFG["analytics_type"]

        self.total_counts = 0
        self.clswise_count = {}

        self.color_cycle = cycle(["#DD00BA", "#042AFF", "#FF4447", "#7D24FF", "#BD00FF"])

        # Set figure size based on image shape
        figsize = (19.2 , 10.8)

        # Ensure line and area chart
        if self.type in {"line", "area"}:
            self.lines = {}
            self.fig = Figure(facecolor=self.bg_color, figsize=figsize)
            self.canvas = FigureCanvas(self.fig)
            self.ax = self.fig.add_subplot(111, facecolor=self.bg_color)
            if self.type == "line":
                (self.line,) = self.ax.plot([], [], color="cyan", linewidth=self.line_width)
        elif self.type in {"bar", "pie"}:
            # Initialize bar or pie plot
            self.fig, self.ax = plt.subplots(figsize=figsize, facecolor=self.bg_color)
            self.canvas = FigureCanvas(self.fig)
            self.ax.set_facecolor(self.bg_color)
            self.color_mapping = {}
            self.ax.axis("equal") if type == "pie" else None  # Ensure pie chart is circular

        # Set common axis properties
        self.ax.set_title(self.title, color=self.fg_color, fontsize=self.fontsize)
        self.ax.set_xlabel(self.x_label, color=self.fg_color, fontsize=self.fontsize - 5)
        self.ax.set_ylabel(self.y_label, color=self.fg_color, fontsize=self.fontsize - 5)
        self.ax.tick_params(axis="both", colors=self.fg_color)

    def process_data(self, im0, frame_number):
        self.extract_tracks(im0)  # Extract tracks

        if self.type == "line":
            for box in self.boxes:
                self.total_counts += 1
            self.update_lines_and_area(frame_number=frame_number)
            self.total_counts = 0
        elif self.type == "pie" or self.type == "bar" or self.type == "area":
            self.clswise_count = {}
            for box, cls in zip(self.boxes, self.clss):
                if self.names[int(cls)] in self.clswise_count:
                    self.clswise_count[self.names[int(cls)]] += 1
                else:
                    self.clswise_count[self.names[int(cls)]] = 1
            if self.type=="area":
                self.update_graph(frame_number=frame_number,
                                  count_dict=self.clswise_count, plot="area")
            if self.type=="bar":
                self.update_graph(frame_number=frame_number,
                                  count_dict=self.clswise_count, plot="bar")

        else:
            Print(f"{self.type} is not supported")
        return im0

    def update_graph(self, frame_number, count_dict=None, plot="line"):
        """
        Update the graph (line or area) with new data for single or multiple classes.

        Args:
            frame_number (int): The current frame number.
            count_dict (dict, optional): Dictionary with class names as keys and counts as values for multiple classes.
                                          If None, updates a single line graph.
            plot (str): Type of the plot i.e. line, bar or area.
        """
        if count_dict is None:
            # Single line update
            x_data = np.append(self.line.get_xdata(), float(frame_number))
            y_data = np.append(self.line.get_ydata(), float(self.total_counts))

            if len(x_data) > self.max_points:
                x_data, y_data = x_data[-self.max_points:], y_data[-self.max_points:]

            self.line.set_data(x_data, y_data)
            self.line.set_label("Counts")
            self.line.set_color("#7b0068")  # Pink color
            self.line.set_marker("*")
            self.line.set_markersize(self.line_width * 5)
        else:
            if plot=="area":
                color_cycle = cycle(["#DD00BA", "#042AFF", "#FF4447", "#7D24FF", "#BD00FF"])
                # Multiple lines or area update
                x_data = self.ax.lines[0].get_xdata() if self.ax.lines else np.array([])
                y_data_dict = {key: np.array([]) for key in count_dict.keys()}

                if self.ax.lines:
                    for line, key in zip(self.ax.lines, count_dict.keys()):
                        y_data_dict[key] = line.get_ydata()

                x_data = np.append(x_data, float(frame_number))
                max_length = len(x_data)

                for key in count_dict.keys():
                    y_data_dict[key] = np.append(y_data_dict[key], float(count_dict[key]))
                    if len(y_data_dict[key]) < max_length:
                        y_data_dict[key] = np.pad(y_data_dict[key], (0, max_length - len(y_data_dict[key])), "constant")

                if len(x_data) > self.max_points:
                    x_data = x_data[1:]
                    for key in count_dict.keys():
                        y_data_dict[key] = y_data_dict[key][1:]

                self.ax.clear()

                for key, y_data in y_data_dict.items():
                    color = next(color_cycle)
                    self.ax.fill_between(x_data, y_data, color=color, alpha=0.7)
                    self.ax.plot(x_data, y_data, color=color, linewidth=self.line_width, marker="o",
                                 markersize=self.line_width * 3, label=f"{key} Data Points")
            if plot=="bar":
                self.ax.clear()     # clear bar data
                labels = list(count_dict.keys())
                counts = list(count_dict.values())
                for label in labels:    # Map labels to colors
                    if label not in self.color_mapping:
                        self.color_mapping[label] = next(self.color_cycle)
                colors = [self.color_mapping[label] for label in labels]
                bars = self.ax.bar(labels, counts, color=colors)
                for bar, count in zip(bars, counts):
                    self.ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        str(count),
                        ha="center",
                        va="bottom",
                        color=self.fg_color,
                    )
                # Create the legend using labels from the bars
                for bar, label in zip(bars, labels):
                    bar.set_label(label)  # Assign label to each bar
                self.ax.legend(loc="upper left", fontsize=13, facecolor=self.fg_color, edgecolor=self.fg_color)

        # Common plot settings
        self.ax.set_facecolor("#f0f0f0")  # Set to light gray or any other color you like
        self.ax.set_title(self.title, color=self.fg_color, fontsize=self.fontsize)
        self.ax.set_xlabel(self.x_label, color=self.fg_color, fontsize=self.fontsize - 3)
        self.ax.set_ylabel(self.y_label, color=self.fg_color, fontsize=self.fontsize - 3)

        # Add and format legend
        legend = self.ax.legend(loc="upper left", fontsize=13, facecolor=self.bg_color, edgecolor=self.bg_color)
        for text in legend.get_texts():
            text.set_color(self.fg_color)

        # Redraw graph, update view, capture, and display the updated plot
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()
        im0 = np.array(self.canvas.renderer.buffer_rgba())
        self.display(im0)

        return im0  # Return the image

    def display(self, im0):
        """
        Write and display the line graph
        Args:
            im0 (ndarray): Image for processing.
        """
        im0 = cv2.cvtColor(im0[:, :, :3], cv2.COLOR_RGBA2BGR)
        self.display_output(im0)

    def update_pie(self, classes_dict):
        """
        Update the pie chart with new data.

        Args:
            classes_dict (dict): Dictionary containing the class data to plot.
        """
        # Update pie chart data
        labels = list(classes_dict.keys())
        sizes = list(classes_dict.values())
        total = sum(sizes)
        percentages = [size / total * 100 for size in sizes]
        start_angle = 90
        self.ax.clear()

        # Create pie chart without labels inside the slices
        wedges, autotexts = self.ax.pie(sizes, autopct=None, startangle=start_angle, textprops={"color": self.fg_color})

        # Construct legend labels with percentages
        legend_labels = [f"{label} ({percentage:.1f}%)" for label, percentage in zip(labels, percentages)]
        self.ax.legend(wedges, legend_labels, title="Classes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

        # Adjust layout to fit the legend
        self.fig.tight_layout()
        self.fig.subplots_adjust(left=0.1, right=0.75)

        # Display and save the updated chart
        im0 = self.fig.canvas.draw()
        im0 = np.array(self.fig.canvas.renderer.buffer_rgba())
        self.display(im0)
