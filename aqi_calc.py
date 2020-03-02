def compute_aqi(pollutant, aqi_dict): 
    
    
    pm2 = {'pm2_good' : (0, 15.4), 'pm2_moderate' : (15.5, 40.4), 'pm2_usg' : (40.5, 65.4), 'pm2_unhealthy' : (65.5, 150.4), 'pm2_veryunhealthy' : (150.5, 250.4), 'pm2_hazardous' : (250.5, 500.4)}
    pm10 = {'pm10_good' : (0, 54), 'pm10_moderate' : (55, 154), 'pm10_usg' : (155, 254), 'pm10_unhealthy' : (255, 354), 'pm10_veryunhealthy' : (355, 424), 'pm10_hazardous' : (425, 604)}
    o3 = {'o3_good' : (0, 0.054), 'o3_moderate' : (0.055, 0.070), 'o3_usg' : (0.071, 0.085), 'o3_unhealthy' : (0.086, 0.105), 'o3_veryunhealthy' : (0.106, 0.200), 'o3_hazardous' : (0.201, 2.00)}
    co = {'co_good' : (0, 4.4), 'co_moderate' : (4.5, 9.4), 'co_usg' : (9.5, 12.4), 'co_unhealthy' : (12.5, 15.4), 'co_veryunhealthy' : (15.5, 30.4), 'co_hazardous' : (30.5, 50.4)}
    no = {'no_good' : (0, 53), 'no_moderate' : (54, 100), 'no_usg' : (101, 360), 'no_unhealthy' : (361, 649), 'no_veryunhealthy' : (650, 1249), 'no_hazardous' : (1250, 2049)}
    so = {'so_good' : (0, 35), 'so_moderate' : (36, 75), 'so_usg' : (76, 185), 'so_unhealthy' : (186, 304), 'so_veryunhealthy' : (305, 604), 'so_hazardous' : (605, 1004)}
    aqis = {'aqi_good' : (0, 50),'aqi_moderate' : (51, 100),'aqi_usg' : (101, 150),'aqi_unhealthy' : (151, 200),'aqi_veryunhealthy' : (201, 300),'aqi_hazardous' : (301, 500)}

    if aqi_dict == 'co':
        aqi_dict_ = co
    elif aqi_dict == 'no':
        aqi_dict_ = no
    elif aqi_dict == 'so':
        aqi_dict_ = so
    elif aqi_dict == 'pm2':
        aqi_dict_ = pm2
    elif aqi_dict == 'pm10':
        aqi_dict_ = pm10
    elif aqi_dict == 'o3':
        aqi_dict_ = o3
    else:
        return 'enter a valid aqi_dict type'

    
    if (pollutant >= aqi_dict_['{}_good'.format(aqi_dict)][0]) & (pollutant <= aqi_dict_['{}_good'.format(aqi_dict)][1]):
        aqi = ((pollutant - aqi_dict_['{}_good'.format(aqi_dict)][0])*(aqis['aqi_good'][1] - aqis['aqi_good'][0])/(aqi_dict_['{}_good'.format(aqi_dict)][1] - aqi_dict_['{}_good'.format(aqi_dict)][0])) + aqis['aqi_good'][0]
    elif (pollutant >= aqi_dict_['{}_moderate'.format(aqi_dict)][0] ) & (pollutant <= aqi_dict_['{}_moderate'.format(aqi_dict)][1]):
        aqi = ((pollutant - aqi_dict_['{}_moderate'.format(aqi_dict)][0])*(aqis['aqi_moderate'][1] - aqis['aqi_moderate'][0])/(aqi_dict_['{}_moderate'.format(aqi_dict)][1] - aqi_dict_['{}_moderate'.format(aqi_dict)][0])) + aqis['aqi_moderate'][0]
    elif (pollutant >= aqi_dict_['{}_usg'.format(aqi_dict)][0] ) & (pollutant <= aqi_dict_['{}_usg'.format(aqi_dict)][1]):
        aqi = ((pollutant - aqi_dict_['{}_usg'.format(aqi_dict)][0])*(aqis['aqi_usg'][1] - aqis['aqi_usg'][0])/(aqi_dict_['{}_usg'.format(aqi_dict)][1] - aqi_dict_['{}_usg'.format(aqi_dict)][0])) + aqis['aqi_usg'][0]
    elif (pollutant >= aqi_dict_['{}_unhealthy'.format(aqi_dict)][0] ) & (pollutant <= aqi_dict_['{}_unhealthy'.format(aqi_dict)][1]):
        aqi = ((pollutant - aqi_dict_['{}_unhealthy'.format(aqi_dict)][0])*(aqis['aqi_unhealthy'][1] - aqis['aqi_unhealthy'][0])/(aqi_dict_['{}_unhealthy'.format(aqi_dict)][1] - aqi_dict_['{}_unhealthy'.format(aqi_dict)][0])) + aqis['aqi_unhealthy'][0]
    elif (pollutant >= aqi_dict_['{}_veryunhealthy'.format(aqi_dict)][0] ) & (pollutant <= aqi_dict_['{}_veryunhealthy'.format(aqi_dict)][1]):
        aqi = ((pollutant - aqi_dict_['{}_veryunhealthy'.format(aqi_dict)][0])*(aqis['aqi_veryunhealthy'][1] - aqis['aqi_veryunhealthy'][0])/(aqi_dict_['{}_veryunhealthy'.format(aqi_dict)][1] - aqi_dict_['{}_veryunhealthy'.format(aqi_dict)][0])) + aqis['aqi_veryunhealthy'][0]
    else: 
        aqi = ((pollutant - aqi_dict_['{}_hazardous'.format(aqi_dict)][0])*(aqis['aqi_hazardous'][1] - aqis['aqi_hazardous'][0])/(aqi_dict_['{}_hazardous'.format(aqi_dict)][1] - aqi_dict_['{}_hazardous'.format(aqi_dict)][0])) + aqis['aqi_hazardous'][0]

    return aqi

#categorize aqi score level i.e. good/moderate/unhealthy etc.

def aqi_level(x):
    if (0 <= x <= 50):
        return "good"
    elif (51 <= x <= 100):
        return  "moderate"
    elif (101 <= x <= 150):
        return "usg"
    elif (151 <= x <= 200):
        return "unhealthy"
    elif (201 <= x <= 300):
        return "very unhealthy"
    elif (301 <= x <= 500):
        return "hazardous"

            
            


