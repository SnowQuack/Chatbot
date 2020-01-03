from pyowm import OWM
import time
OWMKEY = '0f00b648102bb5d3ef307d070eaf9b4e'

owm = OWM(OWMKEY)
obs = owm.weather_at_coords(-33.779254, 151.058792)
loc = obs.get_location()
w = obs.get_weather()
#time = obs.get_reception_time(timeformat='iso')
temp = w.get_temperature(unit='celsius')['temp']
stat = w.get_detailed_status()
suburb = loc.get_name()
ref_time = w.get_reference_time(timeformat='iso')
cur_time = ref_time[:10] + ' ' + str(int(ref_time[11:13])-1) +':' + ref_time[14:16]

print('It is {} degrees celsius today in {} ({} AEDT), the weather status indicates {}'.format(temp, suburb, cur_time, stat))


def get_time():
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(current_time)

print(get_time())