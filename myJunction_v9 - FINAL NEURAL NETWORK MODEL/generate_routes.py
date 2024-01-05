from numpy import random

def generate_routefile():
    random.seed(42)
    steps = 1000
    
    with open("data\myJunction.rou.xml", "w") as routes:
        print("""<routes>
    <vType id="car" vClass="passenger" color="230,230,250"/>
    <vType id="bus" vClass="bus" color="255, 165, 0"/>
    <vType id="trailer" vClass="trailer" color="255,192,203"/>
    <vType id="motorcycle" vClass="motorcycle" color="red"/>
    <vType id="bicycle" vClass="bicycle" color="yellow"/>
    <vType id="mini-truck" vClass="delivery" color="blue"/>
    <vType id="van" vClass="hov" color="green"/>

    <route id="route_01" edges="BWtoW WtoJ JtoE EtoBE BEtoNE NEtoBN"/>
    <route id="route_02" edges="BEtoE EtoJ JtoW WtoBW BWtoNW NWtoBN"/>
    <route id="route_03" edges="BNtoN NtoJ JtoS StoBS BStoSW SWtoBW"/>
    <route id="route_04" edges="BStoS StoJ JtoN NtoBN BNtoNE NEtoBE"/>
    """, file=routes)

        carNum, busNum, truckNum, motorcycleNum, bicycleNum = 0, 0, 0, 0, 0
        r1, r2, r3, r4 = 0, 0, 0, 0
        departTime = 0
        departTimeToggle = True
        intervalTime = 8

        for i in range(steps):
            departTimeToggle = True
            carProb = getGaussianDistribution(0.4, 0.1, 1)
            heavyGoodsVehicleProb = getGaussianDistribution(0.23, 0.1, 1)
            lightweightCommercialVehicleProb = getGaussianDistribution(0.36, 0.1, 1)
            twoWheelerProb = getGaussianDistribution(0.1, 0.1, 1)

            uni_dist = getUniformDistribution(0, 1)
            route = getRouteId("route_01", "route_02", "route_03", "route_04")

            if(carProb >= uni_dist):
                carNum += 1
                if departTimeToggle == True:
                    departTime += intervalTime
                print(f'    <vehicle id="car_{carNum}" type="car" route="{route}" depart="{departTime}" />', file=routes)
                departTimeToggle = False

            if(lightweightCommercialVehicleProb >= uni_dist):
                temp_uni_dist = getUniformDistribution(0, 1)
                if temp_uni_dist < 0.5:
                    truckNum += 1
                    if departTimeToggle == True:
                        departTime += intervalTime
                    print(f'    <vehicle id="truck_{truckNum}" type="mini-truck" route="{route}" depart="{departTime}" />', file=routes)
                    departTimeToggle = False
                else:
                    carNum += 1
                    if departTimeToggle == True:
                        departTime += intervalTime
                    print(f'    <vehicle id="car_{carNum}" type="van" route="{route}" depart="{departTime}" />', file=routes)
                    departTimeToggle = False

            if(heavyGoodsVehicleProb >= uni_dist):
                temp_uni_dist = getUniformDistribution(0, 1)
                if temp_uni_dist <= 0.7:
                    busNum += 1
                    if departTimeToggle == True:
                        departTime += intervalTime
                    print(f'    <vehicle id="bus_{busNum}" type="bus" route="{route}" depart="{departTime}" />', file=routes)
                    departTimeToggle = False
                else:
                    truckNum +=1 
                    if departTimeToggle == True:
                        departTime += intervalTime
                    print(f'    <vehicle id="truck_{truckNum}" type="trailer" route="{route}" depart="{departTime}" />', file=routes)
                    departTimeToggle = False

            if(twoWheelerProb >= uni_dist):
                temp_uni_dist = getUniformDistribution(0, 1)
                if temp_uni_dist < 0.5:
                    bicycleNum += 1
                    if departTimeToggle == True:
                        departTime += intervalTime
                    print(f'    <vehicle id="bicycle_{bicycleNum}" type="bicycle" route="{route}" depart="{departTime}" />', file=routes)
                    departTimeToggle = False
                else:
                    motorcycleNum += 1
                    if departTimeToggle == True:
                        departTime += intervalTime
                    print(f'    <vehicle id="motorcycle_{motorcycleNum}" type="motorcycle" route="{route}" depart="{departTime}" />', file=routes)
                    departTimeToggle = False

        print("</routes>", file=routes)

def getRouteId(route1, route2, route3, route4):
    num = getUniformDistribution(0, 1)
    if num < 0.25:
        return route1
    if num < 0.50:
        return route2
    if num < 0.75:
        return route3
    if num < 1.00:
        return route4
    
def getGaussianDistribution(mean, std, size):
    return random.normal(loc=mean, scale=std, size=size)

def getUniformDistribution(minVal, maxVal):
    return random.uniform(minVal, maxVal)

generate_routefile()