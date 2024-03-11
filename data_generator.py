import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import os


from config import Const, DataShape
# available backends:
# 'GTK3Agg', 'GTK3Cairo', 'GTK4Agg', 'GTK4Cairo', 'MacOSX',
# 'nbAgg', 'QtAgg', 'QtCairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg',
# 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo',
# 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template'
# matplotlib.use('TkAgg')

class NumericalDataGenerator():
    def __init__(self, get_att_only=False) -> None:
        self.data_hasil = []
        self.x_vel_data = []
        self.y_vel_data = []
        self.cell_length = 1/(Const.N_POINTS_Y - 1)
        self.n_point_x = (Const.N_POINTS_Y) * Const.ASPECT_RATIO 
        self.x_range =  np.linspace(0.0, 1.0 * Const.ASPECT_RATIO, self.n_point_x)
        self.y_range = np.linspace(0.0, 1.0, Const.N_POINTS_Y)
        self.coordinates_x , self.coordinates_y = np.meshgrid(self.x_range, self.y_range)
        self.height_var = [num for num in range(8,44)]
        if not get_att_only:
            self.initialMatrix(self.n_point_x, Const.N_POINTS_Y)
            self.initialCondition(Const.STEP_HEIGHT_POINTS, Const.STEP_WIDTH_POINTS)

    def initialMatrix(self, n_points_x, n_points_y):
        """
        Method untuk menginisialisasi matrix dasar yang akan digunakan untuk perhitungan

        @parameter
        *n_points_x : int
        *n_points_y : int
        """
        # Present Velocity
        self.velocity_x_prev = np.ones((n_points_y + 1, n_points_x))
        self.velocity_y_prev = np.zeros((n_points_y, n_points_x+1))

        # Preassure Point
        self.pressure_prev = np.zeros((n_points_y+1, n_points_x+1))

        # Temporary Velocity
        self.velocity_x_tent = np.zeros_like(self.velocity_x_prev)
        self.velocity_y_tent = np.zeros_like(self.velocity_y_prev)

        # Next Velocity
        self.velocity_x_next = np.zeros_like(self.velocity_x_prev)
        self.velocity_y_next = np.zeros_like(self.velocity_y_prev)
    
    def initialCondition(self, step_height, step_width):
        """
        Method yang digunakan untuk menginisialisasi 
        kondisi dasar dari simulasi kedalam matrix dasar

        @parameter
        *step_height : int
        *step_width : int

        """
        self.velocity_x_prev[:step_height + 1, :] = 0.0

        # Top Edge
        self.velocity_x_prev[-1, :] = - self.velocity_x_prev[-2, :]

        # Top Edge Wall
        self.velocity_x_prev[step_height, 1:step_width] = - self.velocity_x_prev[step_height + 1, 1:step_width]

        # Right Edge Wall
        self.velocity_x_prev[1:step_height + 1, step_width] = 0.0

        # Bottom Edge domain
        self.velocity_x_prev[0, step_width + 1: -1] = - self.velocity_x_prev[1, step_width+1: -1]

        # Values inside wall
        self.velocity_x_prev[:step_height, :step_width] = 0.0


    def calculate(self,
        step_height=Const.STEP_HEIGHT_POINTS, 
        step_width=Const.STEP_WIDTH_POINTS, 
        kinematic_viscosity=Const.KINEMATIC_VISCOSITY, 
        time_step_length=Const.TIME_STEP_LENGTH, 
        pressure_poission_itter=Const.N_PRESSURE_POISSON_ITERATIONS, 
        time_step = Const.N_TIME_STEPS,
        plot_every=Const.PLOT_EVERY
    ):
        """
        Method yang digunakan untuk proses kalkulasi numerik, 
        untuk simulasi aliran fluida semi-mampat

        @parameter
        step_height : int
        step_width : int
        kinematic_viscosity : float
        time_step_length : float
        preassure_poission_itter : int
        time_step : int
        plot_every : int

        """
        for iter in tqdm(range(time_step)):
            # Perbaharui nilai dari kecepatan u
            diffusion_x = kinematic_viscosity * (
                (
                    self.velocity_x_prev[1:-1, 2:  ]
                    +
                    self.velocity_x_prev[2:  , 1:-1]
                    +
                    self.velocity_x_prev[1:-1,  :-2]
                    +
                    self.velocity_x_prev[ :-2, 1:-1]
                    - 4 *
                    self.velocity_x_prev[1:-1, 1:-1]
                ) / (
                    self.cell_length**2
                )
            )
            convection_x = (
                (
                    self.velocity_x_prev[1:-1, 2:  ]**2
                    -
                    self.velocity_x_prev[1:-1,  :-2]**2
                ) / (
                    2 * self.cell_length
                )
                +
                (
                    self.velocity_y_prev[1:  , 1:-2]
                    +
                    self.velocity_y_prev[1:  , 2:-1]
                    +
                    self.velocity_y_prev[ :-1, 1:-2]
                    +
                    self.velocity_y_prev[ :-1, 2:-1]
                ) / 4
                *
                (
                    self.velocity_x_prev[2:  , 1:-1]
                    -
                    self.velocity_x_prev[ :-2, 1:-1]
                ) / (
                    2 * self.cell_length
                )
            )
            pressure_gradient_x = (
                (
                    self.pressure_prev[1:-1, 2:-1]
                    -
                    self.pressure_prev[1:-1, 1:-2]
                ) / (
                    self.cell_length
                )
            )

            self.velocity_x_tent[1:-1, 1:-1] = (
                self.velocity_x_prev[1:-1, 1:-1]
                +
                time_step_length
                *
                (
                    -
                    pressure_gradient_x
                    +
                    diffusion_x
                    -
                    convection_x
                )
            )

            # Aplikasikan Kondisi Batas (Boundary Condition)

            # Aliran Masuk (Inflow)
            self.velocity_x_tent[(step_height + 1):-1, 0] = 1.0

            # Aliran Keluar (Outflow)
            inflow_mass_rate_tent = np.sum(self.velocity_x_tent[(step_height + 1):-1, 0])
            outflow_mass_rate_tent = np.sum(self.velocity_x_tent[1:-1, -2])
            self.velocity_x_tent[1:-1, -1] = self.velocity_x_tent[1:-1, -2] * inflow_mass_rate_tent / outflow_mass_rate_tent

            # Ujung Atas Rintangan
            self.velocity_x_tent[step_height, 1:step_width] = - self.velocity_x_tent[step_height + 1, 1:step_width]

            # Ujung kanan Rintangan
            self.velocity_x_tent[1:(step_height + 1), step_width] = 0.0

            # Ujung Bawah rintangan
            self.velocity_x_tent[0, (step_width + 1):-1] =\
                - self.velocity_x_tent[1, (step_width + 1):-1]

            # Ujung Atas pipa
            self.velocity_x_tent[-1, :] = - self.velocity_x_tent[-2, :]

            # Ubah nilai kecepatan menjadi 0 pada bagian dalam rintangan
            self.velocity_x_tent[:step_height, :step_width] = 0.0

            # Perbaharui nilai kecepatan v
            diffusion_y = kinematic_viscosity * (
                (
                    +
                    self.velocity_y_prev[1:-1, 2:  ]
                    +
                    self.velocity_y_prev[2:  , 1:-1]
                    +
                    self.velocity_y_prev[1:-1,  :-2]
                    +
                    self.velocity_y_prev[ :-2, 1:-1]
                    -
                    4 * self.velocity_y_prev[1:-1, 1:-1]
                ) / (
                    self.cell_length**2
                )
            )
            convection_y = (
                (
                    self.velocity_x_prev[2:-1, 1:  ]
                    +
                    self.velocity_x_prev[2:-1,  :-1]
                    +
                    self.velocity_x_prev[1:-2, 1:  ]
                    +
                    self.velocity_x_prev[1:-2,  :-1]
                ) / 4
                *
                (
                    self.velocity_y_prev[1:-1, 2:  ]
                    -
                    self.velocity_y_prev[1:-1,  :-2]
                ) / (
                    2 * self.cell_length
                )
                +
                (
                    self.velocity_y_prev[2:  , 1:-1]**2
                    -
                    self.velocity_y_prev[ :-2, 1:-1]**2
                ) / (
                    2 * self.cell_length
                )
            )
            pressure_gradient_y = (
                (
                    self.pressure_prev[2:-1, 1:-1]
                    -
                    self.pressure_prev[1:-2, 1:-1]
                ) / (
                    self.cell_length
                )
            )

            self.velocity_y_tent[1:-1, 1:-1] = (
                self.velocity_y_prev[1:-1, 1:-1]
                +
                time_step_length
                *
                (
                    -
                    pressure_gradient_y
                    +
                    diffusion_y
                    -
                    convection_y
                )
            )

            # Aplikasikan Kondisi Batas (Boundary Condition)

            # Aliran Masuk (Inflow)
            self.velocity_y_tent[(step_height + 1):-1, 0] = - self.velocity_y_tent[(step_height + 1):-1, 1]

            # Aliran Keluar (Outflow)
            self.velocity_y_tent[1:-1, -1] = self.velocity_y_tent[1:-1, -2]

            # Ujung Atas Rintangan
            self.velocity_y_tent[step_height, 1:(step_width + 1)] = 0.0

            # Ujung Kanan Rintangan
            self.velocity_y_tent[1:(step_height + 1), step_width] = - self.velocity_y_tent[1:(step_height + 1), (step_width + 1)]

            # Ujung Bawah Rintangan
            self.velocity_y_tent[0, (step_width + 1):] = 0.0

            # Ujung Atas pipa
            self.velocity_y_tent[-1, :] = 0.0

            # Ubah nilai kecepatan-v menjadi 0 didalam tepi ringantan
            self.velocity_y_tent[:step_height, :step_width] = 0.0

            # Hitung nilai divergensi dengan ruas kanan akan berupa Pressure Poisson
            divergence = (
                (
                    self.velocity_x_tent[1:-1, 1:  ]
                    -
                    self.velocity_x_tent[1:-1,  :-1]
                ) / (
                    self.cell_length
                )
                +
                (
                    self.velocity_y_tent[1:  , 1:-1]
                    -
                    self.velocity_y_tent[ :-1, 1:-1]
                ) / (
                    self.cell_length
                )
            )
            pressure_poisson_rhs = divergence / time_step_length

            # Selesaikan pressure correction poisson problem
            pressure_correction_prev = np.zeros_like(self.pressure_prev)
            for _ in range(pressure_poission_itter):
                pressure_correction_next = np.zeros_like(pressure_correction_prev)
                pressure_correction_next[1:-1, 1:-1] = 1/4 * (
                    +
                    pressure_correction_prev[1:-1, 2:  ]
                    +
                    pressure_correction_prev[2:  , 1:-1]
                    +
                    pressure_correction_prev[1:-1,  :-2]
                    +
                    pressure_correction_prev[ :-2, 1:-1]
                    -
                    self.cell_length**2
                    *
                    pressure_poisson_rhs
                )

      

                # Aliran Masuk (Inflow)
                pressure_correction_next[(step_height + 1):-1, 0] = pressure_correction_next[(step_height + 1):-1, 1]

                # Aliran Keluar (Outflow)
                pressure_correction_next[1:-1, -1] = - pressure_correction_next[1:-1, -2]

                # Ujung Atas Rintangan
                pressure_correction_next[step_height, 1:(step_width + 1)] = pressure_correction_next[(step_height + 1), 1:(step_width + 1)]

                # Ujung Kanan Rintangan
                pressure_correction_next[1:(step_height + 1), step_width] = pressure_correction_next[1:(step_height + 1), (step_width + 1)]

                # Ujung Bawah pipa
                pressure_correction_next[0, (step_width + 1):-1] =  pressure_correction_next[1, (step_width + 1):-1]

                # Ujung Atas Pipa
                pressure_correction_next[-1, :] = pressure_correction_next[-2, :]

                # Ubah seluruh nilai pressure correction pada bagian dalam rintangan menjadi 0
                pressure_correction_next[:step_height, :step_width] = 0.0

                # Smooting
                pressure_correction_prev = pressure_correction_next

            # Perbaharui nilai tekanan
            pressure_next = self.pressure_prev + pressure_correction_next

            # Incompressibilities
            pressure_correction_gradient_x = (
                (
                    pressure_correction_next[1:-1, 2:-1]
                    -
                    pressure_correction_next[1:-1, 1:-2]
                ) / (
                    self.cell_length
                )
            )

            self.velocity_x_next[1:-1, 1:-1] = (
                self.velocity_x_tent[1:-1, 1:-1]
                -
                time_step_length
                *
                pressure_correction_gradient_x
            )

            pressure_correction_gradient_y = (
                (
                    pressure_correction_next[2:-1, 1:-1]
                    -
                    pressure_correction_next[1:-2, 1:-1]
                ) / (
                    self.cell_length
                )
            )

            self.velocity_y_next[1:-1, 1:-1] = (
                self.velocity_y_tent[1:-1, 1:-1]
                -
                time_step_length
                *
                pressure_correction_gradient_y
            )

            # Aplikasikan Boundary Condition

            # Aliran Masuk (Inflow)
            self.velocity_x_next[(step_height + 1):-1, 0] = 1.0

            # Aliran Keluar (Outflow)
            inflow_mass_rate_next = np.sum(self.velocity_x_next[(step_height + 1):-1, 0])
            outflow_mass_rate_next = np.sum(self.velocity_x_next[1:-1, -2])
            self.velocity_x_next[1:-1, -1] = self.velocity_x_next[1:-1, -2] * inflow_mass_rate_next / outflow_mass_rate_next

            # Ujung Atas Ringantan
            self.velocity_x_next[step_height, 1:step_width] = - self.velocity_x_next[step_height + 1, 1:step_width]

            # Ujung Kanan Rintangan
            self.velocity_x_next[1:(step_height + 1), step_width] = 0.0

            # Ujung Bawah Pipa
            self.velocity_x_next[0, (step_width + 1):-1] = - self.velocity_x_next[1, (step_width + 1):-1]

            # Ujung Atas Pipa
            self.velocity_x_next[-1, :] = - self.velocity_x_next[-2, :]

            # Perbaharui nilai kecepatan u didalam rintangan menjadi 0
            self.velocity_x_next[:step_height, :step_width] = 0.0

            # Aliran Masuk (Inflow)
            self.velocity_y_next[(step_height + 1):-1, 0] = - self.velocity_y_next[(step_height + 1):-1, 1]

            # aliran Keluar (Outflow)
            self.velocity_y_next[1:-1, -1] = self.velocity_y_next[1:-1, -2]

            # Ujung Atas rintangan
            self.velocity_y_next[step_height, 1:(step_width + 1)] = 0.0

            # Ujung Kanan rintangan
            self.velocity_y_next[1:(step_height + 1), step_width] =  - self.velocity_y_next[1:(step_height + 1), (step_width + 1)]

            # Ujung Bawah Pipa
            self.velocity_y_next[0, (step_width + 1):] = 0.0

            # Ujung Atas Pipa
            self.velocity_y_next[-1, :] = 0.0

            # Perbaharui nilai kecepatan-v didalam rintangan menjadi 0
            self.velocity_y_next[:step_height, :step_width] = 0.0

            
            # Perbaharui kondisi untuk waktu selanjutnya
            self.velocity_x_prev = self.velocity_x_next
            self.velocity_y_prev = self.velocity_y_next
            self.pressure_prev = pressure_next

    

            # Visualisasi
            if iter % plot_every == 0:
                velocity_x_vertex_centered = (
                    (
                        self.velocity_x_next[1:  , :]
                        +
                        self.velocity_x_next[ :-1, :]
                    ) / 2
                )
                velocity_y_vertex_centered = (
                    (
                        self.velocity_y_next[:, 1:  ]
                        +
                        self.velocity_y_next[:,  :-1]
                    ) / 2
                )

                velocity_x_vertex_centered[:(step_height + 1),:(step_width + 1)] = 0.0
                velocity_y_vertex_centered[:(step_height + 1),:(step_width + 1)] = 0.0


                self.x_vel_data.append(velocity_x_vertex_centered)
                self.y_vel_data.append(velocity_y_vertex_centered)

    def generator(self):
        """
        Method yang digunakan untuk membuat sebuah generator
        """
        for index in range(len(self.x_vel_data)):
            x_vel_vertex = self.x_vel_data[index]
            y_vel_vertex = self.y_vel_data[index]
            yield x_vel_vertex, y_vel_vertex

    def generateData(self, plot_data=False, save_as=None, combine_axis=False):
        """
        Method yang digunakan sebagai pemantik, 
        untuk menjalankan proses kalkulasi beserta plot data, 
        simpan hasil, dan penggabungan data

        @parameter
        plot_data : boolean
        save_as : string
        combine_axis : boolean

        """
        generator = self.calculate()
        if combine_axis:
            self.data_hasil = np.stack((self.x_vel_data, self.y_vel_data), axis=-1)
        if plot_data:
            generator = self.generator()
            fig, ax = plt.subplots(2,1, figsize=(10, 8))
            plt.pause(0.1)

            def animate(frame):
                velocity_x_vertex_centered, velocity_y_vertex_centered = next(generator)
                ax[0].clear()
                ax[0].contourf(
                    self.coordinates_x,
                    self.coordinates_y,
                    velocity_x_vertex_centered,
                    levels=10,
                    cmap='viridis',
                    vmin=-1.5,
                    vmax=1.5,
                )
                ax[0].quiver(
                    self.coordinates_x[:, ::6],
                    self.coordinates_y[:, ::6],
                    velocity_x_vertex_centered[:, ::6],
                    velocity_y_vertex_centered[:, ::6],
                    alpha=0.4,
                )
                ax[1].clear()
                ax[1].streamplot(
                    self.coordinates_x,
                    self.coordinates_y,
                    velocity_x_vertex_centered,
                    velocity_y_vertex_centered,
                )

            anim = animation.FuncAnimation(fig, animate, frames=DataShape.TIME_STEP)
            # anim = animation.FuncAnimation(fig, animate, frames=100, interval=100, blit=use_blit)
            plt.show()

        if save_as != None:
            self.save(save_as)
        return self.x_vel_data, self.y_vel_data

    def save(self, file_name):
        """
        Method yang digunakan untuk menyimpan 
        hasil perhitunga numerik

        @paramter
        file_name: string
        """
        if not os.path.isdir("data"):
            os.makedirs("data")
        
        np.save(f"data/{file_name}", self.data_hasil)
        print("Data Saved !")


