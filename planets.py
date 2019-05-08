#Planetary database
#Source for planetary data, and some of the data on
#the moons, is http://nssdc.gsfc.nasa.gov/planetary/factsheet/

class Planet:
    '''
    A Planet object contains basic planetary data.
    If P is a Planet object, the data are:
           P.name = Name of the planet
           P.a = Mean radius of planet (m)
           P.g = Surface gravitational acceleration (m/s**2)
           P.L = Annual mean solar constant (current) (W/m**2)
           P.albedo Bond albedo (fraction)
           
           P.rsm = Semi-major axis of orbit about Sun (m)
           P.year = Sidereal length of year (s)
           P.eccentricity =  Eccentricity (unitless)
           P.day = Mean tropical length of day (s)
           P.obliquity = Obliquity to orbit (degrees)
           P.Lequinox = Longitude of equinox (degrees)

           P.Tsbar = Mean surface temperature (K)
           P.Tsmax = Maximum surface temperature (K)

    For gas giants, "surface" quantities are given at the 1 bar level
    '''

    #__repr__ object prints out a help string when help is
    #invoked on the planet object or the planet name is typed
    def __repr__(self):
        line1 =\
        'This planet object contains information on %s\n'%self.name
        line2 = 'Type \"help(Planet)\" for more information\n'
        return line1+line2
    def __init__(self):
        self.name = None #Name of the planet
        self.a = None #Mean radius of planet
        self.g = None #Surface gravitational acceleration
        self.L = None #Annual mean solar constant (current)
        self.albedo = None #Bond albedo
        
        self.rsm = None #Semi-major axis
        self.year = None #Sidereal length of year
        self.eccentricity = None # Eccentricity
        self.day = None #Mean tropical length of day
        self.obliquity = None #Obliquity to orbit
        self.Lequinox = None #Longitude of equinox

        self.Tsbar = None #Mean surface temperature
        self.Tsmax = None #Maximum surface temperature

#----------------------------------------------------       
Mercury = Planet()        
Mercury.name = 'Mercury' #Name of the planet
Mercury.a = 2.4397e6 #Mean radius of planet
Mercury.g = 3.70 #Surface gravitational acceleration
Mercury.albedo = .119 #Bond albedo
Mercury.L = 9126.6 #Annual mean solar constant (current)
#
Mercury.rsm = 57.91e9 #Semi-major axis
Mercury.year = 87.969*24.*3600. #Sidereal length of year
Mercury.eccentricity = .2056 # Eccentricity
Mercury.day = 4222.6*3600. #Mean tropical length of day
Mercury.obliquity = .01 #Obliquity to orbit (deg)
Mercury.Lequinox = None #Longitude of equinox (deg)
#
Mercury.Tsbar = 440. #Mean surface temperature
Mercury.Tsmax = 725. #Maximum surface temperature

#----------------------------------------------------        
Venus = Planet()
Venus.name = 'Venus' #Name of the planet
Venus.a = 6.0518e6 #Mean radius of planet
Venus.g = 8.87 #Surface gravitational acceleration
Venus.albedo = .750 #Bond albedo
Venus.L = 2613.9 #Annual mean solar constant (current)
#
Venus.rsm = 108.21e9 #Semi-major axis
Venus.year = 224.701*24.*3600. #Sidereal length of year
Venus.eccentricity = .0067 # Eccentricity
Venus.day = 2802.*3600. #Mean tropical length of day
Venus.obliquity = 177.36 #Obliquity to orbit (deg)
Venus.Lequinox = None #Longitude of equinox (deg)
#
Venus.Tsbar = 737. #Mean surface temperature
Venus.Tsmax = 737. #Maximum surface temperature

#----------------------------------------------------        
Earth = Planet()
Earth.name = 'Earth' #Name of the planet
Earth.a = 6.371e6 #Mean radius of planet
Earth.g = 9.798 #Surface gravitational acceleration
Earth.albedo = .306 #Bond albedo
Earth.L = 1367.6 #Annual mean solar constant (current)
#
Earth.rsm = 149.60e9 #Semi-major axis
Earth.year = 365.256*24.*3600. #Sidereal length of year
Earth.eccentricity = .0167 # Eccentricity
Earth.day = 24.000*3600. #Mean tropical length of day
Earth.obliquity = 23.45 #Obliquity to orbit (deg)
Earth.Lequinox = None #Longitude of equinox (deg)
#
Earth.Tsbar = 288. #Mean surface temperature
Earth.Tsmax = None #Maximum surface temperature

#----------------------------------------------------        
Mars = Planet()
Mars.name = 'Mars' #Name of the planet
Mars.a = 3.390e6 #Mean radius of planet
Mars.g = 3.71 #Surface gravitational acceleration
Mars.albedo = .250 #Bond albedo
Mars.L = 589.2 #Annual mean solar constant (current)
#
Mars.rsm = 227.92e9 #Semi-major axis
Mars.year = 686.98*24.*3600. #Sidereal length of year
Mars.eccentricity = .0935 # Eccentricity
Mars.day = 24.6597*3600. #Mean tropical length of day
Mars.obliquity = 25.19 #Obliquity to orbit (deg)
Mars.Lequinox = None #Longitude of equinox (deg)
#
Mars.Tsbar = 210. #Mean surface temperature
Mars.Tsmax = 295. #Maximum surface temperature

#----------------------------------------------------        
Jupiter = Planet()
Jupiter.name = 'Jupiter' #Name of the planet
Jupiter.a = 69.911e6 #Mean radius of planet
Jupiter.g = 24.79 #Surface gravitational acceleration
Jupiter.albedo = .343 #Bond albedo
Jupiter.L = 50.5 #Annual mean solar constant (current)
#
Jupiter.rsm = 778.57e9 #Semi-major axis
Jupiter.year = 4332.*24.*3600. #Sidereal length of year
Jupiter.eccentricity = .0489 # Eccentricity
Jupiter.day = 9.9259*3600. #Mean tropical length of day
Jupiter.obliquity = 3.13 #Obliquity to orbit (deg)
Jupiter.Lequinox = None #Longitude of equinox (deg)
#
Jupiter.Tsbar = 165. #Mean surface temperature
Jupiter.Tsmax = None #Maximum surface temperature

#----------------------------------------------------        
Saturn = Planet()
Saturn.name = 'Saturn' #Name of the planet
Saturn.a = 58.232e6 #Mean radius of planet
Saturn.g = 10.44 #Surface gravitational acceleration
Saturn.albedo = .342 #Bond albedo
Saturn.L = 14.90 #Annual mean solar constant (current)
#
Saturn.rsm = 1433.e9 #Semi-major axis
Saturn.year = 10759.*24.*3600. #Sidereal length of year
Saturn.eccentricity = .0565 # Eccentricity
Saturn.day = 10.656*3600. #Mean tropical length of day
Saturn.obliquity = 26.73 #Obliquity to orbit (deg)
Saturn.Lequinox = None #Longitude of equinox (deg)
#
Saturn.Tsbar = 134. #Mean surface temperature
Saturn.Tsmax = None #Maximum surface temperature

#----------------------------------------------------        
Uranus = Planet()
Uranus.name = 'Uranus' #Name of the planet
Uranus.a = 25.362e6 #Mean radius of planet
Uranus.g = 8.87 #Surface gravitational acceleration
Uranus.albedo = .300 #Bond albedo
Uranus.L = 3.71 #Annual mean solar constant (current)
#
Uranus.rsm = 2872.46e9 #Semi-major axis
Uranus.year = 30685.4*24.*3600. #Sidereal length of year
Uranus.eccentricity = .0457 # Eccentricity
Uranus.day = 17.24*3600. #Mean tropical length of day
Uranus.obliquity = 97.77 #Obliquity to orbit (deg)
Uranus.Lequinox = None #Longitude of equinox (deg)
#
Uranus.Tsbar = 76. #Mean surface temperature
Uranus.Tsmax = None #Maximum surface temperature


#----------------------------------------------------        
Neptune = Planet()
Neptune.name = 'Neptune' #Name of the planet
Neptune.a = 26.624e6 #Mean radius of planet
Neptune.g = 11.15 #Surface gravitational acceleration
Neptune.albedo = .290 #Bond albedo
Neptune.L = 1.51 #Annual mean solar constant (current)
#
Neptune.rsm = 4495.06e9 #Semi-major axis
Neptune.year = 60189.0*24.*3600. #Sidereal length of year
Neptune.eccentricity = .0113 # Eccentricity
Neptune.day = 16.11*3600. #Mean tropical length of day
Neptune.obliquity = 28.32 #Obliquity to orbit (deg)
Neptune.Lequinox = None #Longitude of equinox (deg)
#
Neptune.Tsbar = 72. #Mean surface temperature
Neptune.Tsmax = None #Maximum surface temperature

#----------------------------------------------------        
Pluto = Planet()
Pluto.name = 'Pluto' #Name of the planet
Pluto.a = 1.195e6 #Mean radius of planet
Pluto.g = .58 #Surface gravitational acceleration
Pluto.albedo = .5 #Bond albedo
Pluto.L = .89 #Annual mean solar constant (current)
#
Pluto.rsm = 5906.e9 #Semi-major axis
Pluto.year = 90465.*24.*3600. #Sidereal length of year
Pluto.eccentricity = .2488 # Eccentricity
Pluto.day = 153.2820*3600. #Mean tropical length of day
Pluto.obliquity = 122.53 #Obliquity to orbit (deg)
Pluto.Lequinox = None #Longitude of equinox (deg)
#
Pluto.Tsbar = 50. #Mean surface temperature
Pluto.Tsmax = None #Maximum surface temperature



#Selected moons

#----------------------------------------------------        
Moon = Planet()
Moon.name = 'Moon' #Name of the planet
Moon.a = 1.737e6 #Mean radius of planet
Moon.g = 1.62 #Surface gravitational acceleration
Moon.albedo = .11 #Bond albedo
Moon.L = 1367.6 #Annual mean solar constant (current)
#
Moon.rsm = Earth.rsm #Semi-major axis
Moon.year = Earth.year #Sidereal length of year
Moon.eccentricity = None # Eccentricity
Moon.day = 28.*24.*3600. #Mean tropical length of day (approx)
Moon.obliquity = None #Obliquity to orbit (deg)
Moon.Lequinox = None #Longitude of equinox (deg)
#
Moon.Tsbar = None #Mean surface temperature
Moon.Tsmax = 400. #Maximum surface temperature
Moon.Tsmin = 100. #Minimum surface temperature

Titan = Planet()
Titan.name = 'Titan' #Name of the planet
Titan.a = 2.575e6 #Mean radius of planet
Titan.g = 1.35 #Surface gravitational acceleration
Titan.L = Saturn.L #Annual mean solar constant (current)
Titan.albedo = .21 #Bond albedo (Not yet updated from Cassini)
#        
Titan.rsm = None #Semi-major axis
Titan.year = Saturn.year #Sidereal length of year
Titan.eccentricity = Saturn.eccentricity # Eccentricity ABOUT SUN
Titan.day = 15.9452*24.*3600. #Mean tropical length of day
Titan.obliquity = Saturn.obliquity #Obliquity to plane of Ecliptic
                                   #(Titan's rotation axis approx parallel
                                   # to Saturn's
Titan.Lequinox = Saturn.Lequinox #Longitude of equinox
#
Titan.Tsbar = 95. #Mean surface temperature
Titan.Tsmax = None #Maximum surface temperature

Europa = Planet()
Europa.name = 'Europa' #Name of the planet
Europa.a = 1.560e6 #Mean radius of planet
Europa.g = 1.31 #Surface gravitational acceleration
Europa.L = Jupiter.L #Annual mean solar constant (current)
Europa.albedo = .67 #Bond albedo
#        
Europa.rsm = Jupiter.rsm #Semi-major axis
Europa.year = Jupiter.year #Sidereal length of year
Europa.eccentricity = Jupiter.eccentricity # Eccentricity
Europa.day = 3.551*24.*3600. #Mean tropical length of day
Europa.obliquity = Jupiter.obliquity #Obliquity to plane of ecliptic
Europa.Lequinox = None #Longitude of equinox
#
Europa.Tsbar = 103. #Mean surface temperature
Europa.Tsmax = 125. #Maximum surface temperature

Triton = Planet()
Triton.name = 'Triton' #Name of the planet
Triton.a = 2.7068e6/2. #Mean radius of planet
Triton.g = .78 #Surface gravitational acceleration
Triton.L = Neptune.L #Annual mean solar constant (current)
Triton.albedo = .76 #Bond albedo
#        
Triton.rsm = Neptune.rsm #Semi-major axis
Triton.year = Neptune.year #Sidereal length of year
Triton.eccentricity = Neptune.eccentricity # Eccentricity about Sun
Triton.day = 5.877*24.*3600. #Mean tropical length of day
                             #Triton's rotation is retrograde
Triton.obliquity = 156. #Obliquity to ecliptic **ToDo: Check this.
                        #Note: Seasons are influenced by the inclination
                        #of Triton's orbit? (About 20 degrees to
                        #Neptune's equator
Triton.Lequinox = None #Longitude of equinox
#
Triton.Tsbar = 34.5 #Mean surface temperature
                    #This is probably a computed blackbody
                    #temperature, rather than an observation
Triton.Tsmax = None #Maximum surface temperature


