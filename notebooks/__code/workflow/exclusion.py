import logging
import numpy as np
from IPython.display import display, HTML
import ipywidgets as widgets
import plotly.graph_objects as go
from plotly.offline import iplot
from ipywidgets import interactive

from __code.parent import Parent
from __code import DataType


class Exclusion(Parent):

    def selection_mode(self):
        """Exclusion selection mode."""
        display(HTML("<h3>Exclusion of images from the reconstruction:</h3>"))
        self.exclusion_mode_widget = widgets.RadioButtons(options=["Manual input image index to exclude",
                                                                   "All integrated images counts below a given threshold"],
                                                        layout=widgets.Layout(width='80%'),
                                                        value="Manual input image index to exclude")
        display(self.exclusion_mode_widget)

    def process_exclusion_mode(self):
        """Process the exclusion mode selected by the user."""
        if self.exclusion_mode_widget.value == "Manual input image index to exclude":
            self.list_of_images_to_exclude()
        elif self.exclusion_mode_widget.value == "All integrated images counts below a given threshold":
            self.exclude_images_below_threshold()
        else:
            logging.error("Unknown exclusion mode selected.")

    def list_of_images_to_exclude(self):
        """List of images to exclude from the reconstruction."""
        label = widgets.HTML("<b>Define the index of the images you want to exclude from the reconstruction:</b>")
        display(label)
        self.list_index_widget = widgets.Text(
            value='',
            layout=widgets.Layout(width='80%'),
            placeholder='')
        display(self.list_index_widget)

        example = widgets.HTML("<font color='gray' size='2'><i>Example: 0, 1, 2, 10-20 (to exclude images with index 0, 1, 2 and from 10 to 20)</i></font>")
        display(example)

    def exclude_images_below_threshold(self):
    
        sample_data = self.parent.master_3d_data_array[DataType.sample]
        integrated_intensity = sample_data.sum(axis=(1,2))

        display(HTML("<b>Use the slider to define the threshold below which images will be excluded from the reconstruction:</b>"))

        def on_threshold_change(threshold_value):

            list_index_below_threshold = np.where(integrated_intensity < threshold_value)[0].tolist() 
            list_intensity_below_threshold = integrated_intensity[list_index_below_threshold]

            # Create plotly figure
            fig = go.Figure()

            # Add all data points (green stars)
            fig.add_trace(go.Scatter(
                x=list(range(len(integrated_intensity))),
                y=integrated_intensity,
                mode='markers',
                marker=dict(color='green', symbol='star', size=8),
                name='All images',
                hovertemplate='Index: %{x}<br>Intensity: %{y:.2f}<extra></extra>'
            ))

            # Add points below threshold (red circles)
            if list_index_below_threshold:
                fig.add_trace(go.Scatter(
                    x=list_index_below_threshold,
                    y=list_intensity_below_threshold,
                    mode='markers',
                    marker=dict(color='red', symbol='circle', size=8),
                    name='Below threshold',
                    hovertemplate='Index: %{x}<br>Intensity: %{y:.2f}<extra></extra>'
                ))

            # Add threshold line
            fig.add_hline(
                y=threshold_value,
                line_dash="dash",
                line_color="blue",
                annotation_text=f'Threshold = {threshold_value:.1f}',
                annotation_position="top right"
            )

            # Update layout
            fig.update_layout(
                title='Integrated intensity (full image) vs image index',
                xaxis_title='Image index',
                yaxis_title='Integrated intensity (a.u.)',
                width=700,
                height=500,
                showlegend=True
            )

            fig.show()

            return list_index_below_threshold

        self.display_threshold = interactive(on_threshold_change,
                                        threshold_value=widgets.FloatSlider(
                                                    value=integrated_intensity.min(),
                                                    min=integrated_intensity.min(),
                                                    max=integrated_intensity.max(),
                                                    step=(integrated_intensity.max() - integrated_intensity.min()) / 100,
                                                    description='Threshold:',
                                                    continuous_update=False,
                                                    orientation='horizontal',
                                                    layout=widgets.Layout(width='80%')
                                        ),
        )

        display(self.display_threshold)

    def exclude_this_list_of_images(self):

        if self.exclusion_mode_widget.value == "All integrated images counts below a given threshold":
            list_of_images_to_exclude = self.display_threshold.result
        else:
            list_of_images_to_exclude = self.get_list_of_images_to_exclude()

        # update master_3d_data_array to exclude the images, list_of_images, final_list_of_angles, final_list_of_angles_rad
        if list_of_images_to_exclude:

            logging.info(f"Before exclusion:")
            logging.info(f"\t{len(self.parent.list_of_images[DataType.sample])} images remain for reconstruction.")
            logging.info(f"\t{len(self.parent.final_list_of_angles)} angles remain for reconstruction.")
            logging.info(f"\t{len(self.parent.final_list_of_angles_rad)} angles (rad) remain for reconstruction.")

            logging.info(f"Excluding images with index: {list_of_images_to_exclude} from the reconstruction.")
            mask = np.ones(len(self.parent.list_of_images[DataType.sample]), dtype=bool)
            mask[list_of_images_to_exclude] = False

            self.parent.master_3d_data_array[DataType.sample] = self.parent.master_3d_data_array[DataType.sample][mask]
            self.parent.list_of_images[DataType.sample] = [img for idx, img in enumerate(self.parent.list_of_images[DataType.sample]) if mask[idx]]
            self.parent.final_list_of_angles = [angle for idx, angle in enumerate(self.parent.final_list_of_angles) if mask[idx]]
            self.parent.final_list_of_angles_rad = [angle for idx, angle in enumerate(self.parent.final_list_of_angles_rad) if mask[idx]]

            logging.info(f"After exclusion:")
            logging.info(f"\t{len(self.parent.list_of_images[DataType.sample])} images remain for reconstruction.")
            logging.info(f"\t{len(self.parent.final_list_of_angles)} angles remain for reconstruction.")
            logging.info(f"\t{len(self.parent.final_list_of_angles_rad)} angles (rad) remain for reconstruction.")

    def get_list_of_images_to_exclude(self):
        """Get the list of images to exclude from the reconstruction."""
        list_index_str = self.list_index_widget.value
        list_index = []
        if list_index_str:
            parts = list_index_str.split(',')
            for part in parts:
                part = part.strip()
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    list_index.extend(range(start, end + 1))
                else:
                    list_index.append(int(part))
        logging.info(f"List of images to exclude: {list_index}")
        return list_index
