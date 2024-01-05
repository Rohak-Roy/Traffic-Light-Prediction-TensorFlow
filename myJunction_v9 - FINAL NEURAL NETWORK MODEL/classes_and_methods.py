import traci
import numpy as np
import matplotlib.pyplot as plt

timeForPredictingVehAtTLS = 10

class EdgeInfo:
    def __init__(self):
        self.vehiclesStopped = 0
        self.totalNumOfVeh = 0
        self.CO2Emission = 0
        self.waitingTime = 0
        self.carInfo = {"Number": 0, "Mean Speed": 0, "Mean Distance from Traffic Light": 0}
        self.busInfo = {"Number": 0, "Mean Speed": 0, "Mean Distance from Traffic Light": 0}
        self.truckInfo = {"Number": 0, "Mean Speed": 0, "Mean Distance from Traffic Light": 0}
        self.motorcycleInfo = {"Number": 0, "Mean Speed": 0, "Mean Distance from Traffic Light": 0}
        self.bicycleInfo = {"Number": 0, "Mean Speed": 0, "Mean Distance from Traffic Light": 0}
        self.predictedVehiclesAtTLS = {"Time": 0, "Number": 0}

class LaneInfo:
    def __init__(self):
        self.vehiclesStopped = 0
        self.totalNumOfVeh = 0
        self.CO2Emission = 0
        self.waitingTime = 0
        self.carInfo = {"Number": 0, "Mean Speed": 0, "Mean Distance from Traffic Light": 0}
        self.busInfo = {"Number": 0, "Mean Speed": 0, "Mean Distance from Traffic Light": 0}
        self.truckInfo = {"Number": 0, "Mean Speed": 0, "Mean Distance from Traffic Light": 0}
        self.motorcycleInfo = {"Number": 0, "Mean Speed": 0, "Mean Distance from Traffic Light": 0}
        self.bicycleInfo = {"Number": 0, "Mean Speed": 0, "Mean Distance from Traffic Light": 0}

class WeightInfo:
    def __init__(self):
        self.vehiclesStopped = 0
        self.waitingTime = 0
        self.CO2Emission = 0
        self.vehicles = {"Car": 0, "Bus": 0, "Truck": 0, "Motorcycle": 0, "Bicycle": 0}
        self.predictedVehiclesAtTLS = 0
        self.totalWeight = 0

def getVehicleInfo(allVehicles, vehicleType):
    vehicles =  [item for item in allVehicles if vehicleType in item]
    vehicleSpeeds = list(map(traci.vehicle.getSpeed, vehicles))
    meanVehicleSpeed = sum(vehicleSpeeds) / len(vehicles) if len(vehicles) != 0 else 0
    nextTLS = list(map(traci.vehicle.getNextTLS, vehicles))                 # nextTLS is of type [((...), (...)), ((...), (...))]

    if len(nextTLS) != 0:
        distanceFromTLS = [item[0][2] for item in nextTLS]
    else:
        distanceFromTLS = []

    meanDistanceFromTLS = sum(distanceFromTLS) / len(distanceFromTLS) if len(distanceFromTLS) != 0 else 0

    return {"Number": len(vehicles), "Mean Speed": meanVehicleSpeed, "Mean Distance from Traffic Light": meanDistanceFromTLS}

def getEdgeInfo(edgeID):
    allVehicles = traci.edge.getLastStepVehicleIDs(edgeID)
    edgeInfo = EdgeInfo()

    edgeInfo.vehiclesStopped = traci.edge.getLastStepHaltingNumber(edgeID)
    edgeInfo.totalNumOfVeh = traci.edge.getLastStepVehicleNumber(edgeID)
    edgeInfo.CO2Emission = traci.edge.getCO2Emission(edgeID)
    edgeInfo.waitingTime = traci.edge.getWaitingTime(edgeID)

    predictedVehiclesAtTLSInfo = getNumOfPredictedVehicles(edgeID, timeForPredictingVehAtTLS)
    edgeInfo.predictedVehiclesAtTLS.update(predictedVehiclesAtTLSInfo)

    carInfo = getVehicleInfo(allVehicles, 'car')
    edgeInfo.carInfo.update(carInfo)

    busInfo = getVehicleInfo(allVehicles, 'bus')
    edgeInfo.busInfo.update(busInfo)

    truckInfo = getVehicleInfo(allVehicles, 'truck')
    edgeInfo.truckInfo.update(truckInfo)

    motorcycleInfo = getVehicleInfo(allVehicles, 'motorcycle')
    edgeInfo.motorcycleInfo.update(motorcycleInfo)

    bicycleInfo = getVehicleInfo(allVehicles, 'bicycle')
    edgeInfo.bicycleInfo.update(bicycleInfo)

    return edgeInfo

def printEdgeInfo(edgeInfo, title=""):
    print(f'\n{title}')
    print(f'Number of vehicles currently stopped at traffic light= {edgeInfo.vehiclesStopped}')
    print(f'Total number of vehicles = {edgeInfo.totalNumOfVeh}')
    print(f'Total waiting time of vehicles = {edgeInfo.waitingTime}')
    print(f'Car Info = {edgeInfo.carInfo}')
    print(f'Bus Info = {edgeInfo.busInfo}')
    print(f'Truck Info = {edgeInfo.truckInfo}')
    print(f'Motorcycle Info = {edgeInfo.motorcycleInfo}')
    print(f'Bicycle Info = {edgeInfo.bicycleInfo}')
    print(f'Carbon Emissions released = {edgeInfo.CO2Emission}')
    print(f'Number of vehicles predicted at traffic light after {edgeInfo.predictedVehiclesAtTLS["Time"]}s = {edgeInfo.predictedVehiclesAtTLS["Number"]}')

def getLaneInfo(laneID):
    allVehicles = traci.lane.getLastStepVehicleIDs(laneID)
    laneInfo = LaneInfo()

    laneInfo.vehiclesStopped = traci.lane.getLastStepHaltingNumber(laneID)
    laneInfo.totalNumOfVeh = traci.lane.getLastStepVehicleNumber(laneID)
    laneInfo.CO2Emission = traci.lane.getCO2Emission(laneID)
    laneInfo.waitingTime = traci.lane.getWaitingTime(laneID)

    carInfo = getVehicleInfo(allVehicles, 'car')
    laneInfo.carInfo.update(carInfo)

    busInfo = getVehicleInfo(allVehicles, 'bus')
    laneInfo.busInfo.update(busInfo)

    truckInfo = getVehicleInfo(allVehicles, 'truck')
    laneInfo.truckInfo.update(truckInfo)

    motorcycleInfo = getVehicleInfo(allVehicles, 'motorcycle')
    laneInfo.motorcycleInfo.update(motorcycleInfo)

    bicycleInfo = getVehicleInfo(allVehicles, 'bicycle')
    laneInfo.bicycleInfo.update(bicycleInfo)

    return laneInfo

def printSingleLaneInfo(laneInfo, title=""):
    print(f'\n{title}')
    print(f'Number of vehicles currently stopped at traffic light= {laneInfo.vehiclesStopped}')
    print(f'Total number of vehicles = {laneInfo.totalNumOfVeh}')
    print(f'Total waiting time of vehicles = {laneInfo.waitingTime}')
    print(f'Car Info = {laneInfo.carInfo}')
    print(f'Bus Info = {laneInfo.busInfo}')
    print(f'Truck Info = {laneInfo.truckInfo}')
    print(f'Motorcycle Info = {laneInfo.motorcycleInfo}')
    print(f'Bicycle Info = {laneInfo.bicycleInfo}')
    print(f'Carbon Emissions released = {laneInfo.CO2Emission}')

def printAllLanesInfo(laneIDs, direction=""):
    for idx, laneID in enumerate(laneIDs):
        lane = getLaneInfo(laneID)
        printSingleLaneInfo(lane, f'Information about Lane {idx + 1} in direction {direction}.')

def getLaneIDsFromEdgeID(edgeID):
    laneIDs = []
    numOfLanes = traci.edge.getLaneNumber(edgeID)
    
    for idx in range(numOfLanes):
        laneID = edgeID + f'_{idx}'
        laneIDs.append(laneID)

    return laneIDs

# An exponential graph where f(4) = 2, f(6) = 4, f(8) = 8, f(10) = 16, f(12) = 32 and so on.
def exponential(x, vertical_stretch=1/2, horizontal_stretch=1/2.88539008178, coefficient=1):
    y = coefficient * (vertical_stretch * np.exp(horizontal_stretch * x))
    return y

def decayForTimeSinceTLSChange(x):
    y = 30 * (np.exp(-1/3.69269373069 * x))
    return y

def logarithm(x):
    y = np.log2(x + 1)
    return y

def getNumOfPredictedVehicles(edgeID, time):
    allVehicles = traci.edge.getLastStepVehicleIDs(edgeID)

    if len(allVehicles) != 0:
        nextTLS = list(map(traci.vehicle.getNextTLS, allVehicles))
        distancesFromTLS = [item[0][2] for item in nextTLS]
        vehicleSpeeds = list(map(traci.vehicle.getSpeed, allVehicles))
        filteredVehicleSpeeds = [item for item in vehicleSpeeds if item > 3]
        distanceCoveredInGivenTime = [item * time for item in filteredVehicleSpeeds]
        predictedDistanceFromTLS = [(x - y) for x,y in zip(distancesFromTLS, distanceCoveredInGivenTime)]
        numOfPredictedVehiclesAtTLS = len([item for item in predictedDistanceFromTLS if item < 1.1])
        
        return {"Time": time, "Number": numOfPredictedVehiclesAtTLS}

    return {"Time": time, "Number": 0}

def getWeightInfo(edgeInfo1, edgeInfo2):
    weight = WeightInfo()

    numOfVehiclesStopped = edgeInfo1.vehiclesStopped + edgeInfo2.vehiclesStopped
    totalWaitingTime = edgeInfo1.waitingTime + edgeInfo2.waitingTime
    totalCO2Emission = edgeInfo1.CO2Emission + edgeInfo2.CO2Emission
    numOfCars = edgeInfo1.carInfo["Number"] + edgeInfo2.carInfo["Number"]
    numOfBuses = edgeInfo1.busInfo["Number"] + edgeInfo2.busInfo["Number"]
    numOfTrucks = edgeInfo1.truckInfo["Number"] + edgeInfo2.truckInfo["Number"]
    numOfMotorcycles = edgeInfo1.motorcycleInfo["Number"] + edgeInfo2.motorcycleInfo["Number"]
    numOfBicycles = edgeInfo1.bicycleInfo["Number"] + edgeInfo2.bicycleInfo["Number"]
    numOfPredictedVehiclesAtTLS = edgeInfo1.predictedVehiclesAtTLS["Number"] + edgeInfo2.predictedVehiclesAtTLS["Number"]

    weight.vehiclesStopped = logarithm(numOfVehiclesStopped)
    weight.waitingTime = exponential(totalWaitingTime, horizontal_stretch=1/50)
    weight.CO2Emission = logarithm(totalCO2Emission)
    weight.predictedVehiclesAtTLS = exponential(numOfPredictedVehiclesAtTLS, vertical_stretch=2) if numOfPredictedVehiclesAtTLS != 0 else 0

    weight.vehicles["Car"] = logarithm(numOfCars)
    weight.vehicles["Bus"] = exponential(numOfBuses, 0.45)
    weight.vehicles["Truck"] = exponential(numOfTrucks, 0.45)
    weight.vehicles["Motorcycles"] = exponential(numOfMotorcycles, 0.25)
    weight.vehicles["Bicycles"] = exponential(numOfBicycles, 0.125)
    weight.totalWeight = getTotalWeight(weight)

    return weight

def getTotalWeight(weightInfo):
    totalWeight = weightInfo.vehiclesStopped + weightInfo.waitingTime + weightInfo.CO2Emission + weightInfo.predictedVehiclesAtTLS
    
    for item in weightInfo.vehicles:
        totalWeight += weightInfo.vehicles[item]

    return totalWeight

def printWeightInfo(weightInfo, title=""):
    print(f'\n{title}')
    print(f'Weight due to number of vehicles stopped at traffic light = {weightInfo.vehiclesStopped}')
    print(f'Weight due to waiting time of vehicles stopped = {weightInfo.waitingTime}')
    print(f'Weight due to carbon emissions = {weightInfo.CO2Emission}')
    print(f'Weight due to number of cars = {weightInfo.vehicles["Car"]}')
    print(f'Weight due to number of buses = {weightInfo.vehicles["Bus"]}')
    print(f'Weight due to number of trucks = {weightInfo.vehicles["Truck"]}')
    print(f'Weight due to number of motorcycles = {weightInfo.vehicles["Motorcycle"]}')
    print(f'Weight due to number of bicycles = {weightInfo.vehicles["Bicycle"]}')
    print(f'Weight due to number of vehicles predicted at traffic light = {weightInfo.predictedVehiclesAtTLS}')
    print(f'Total Weight = {weightInfo.totalWeight}')

def displayCumulative(dataFrame1, dataFrame2, title):
    y1, y2 = [], []
    j1, j2 = 0, 0
    fig = plt.figure()

    for idx in range(len(dataFrame1)):
        j1 += dataFrame1[idx]
        y1.append(j1)

    for idx in range(len(dataFrame2)):
        j2 += dataFrame2[idx]
        y2.append(j2)

    x1 = np.arange(0, len(y1))
    x2 = np.arange(0, len(y2))

    fig.add_subplot(1, 2, 1)
    plt.plot(x1, y1)
    plt.xlabel('Seconds Elapsed.')
    plt.ylabel(f'Cumulative {title} - Before.')

    fig.add_subplot(1, 2, 2)
    plt.plot(x2, y2)
    plt.xlabel('Seconds Elapsed.')
    plt.ylabel(f'Cumulative {title} - After.')
    
    plt.show()

def percentChange(dataFrame_before, dataFrame_after, columnName):
    sum1 = dataFrame_before[columnName].sum()
    sum2 = dataFrame_after[columnName].sum()
    difference = sum1 - sum2
    percentChange = (difference/sum1) * 100

    return round(percentChange, 2)

def getData(edgeInfo1, edgeInfo2, edgeInfo3, edgeInfo4):
    numOfVehiclesStopped_horizontal = edgeInfo1.vehiclesStopped + edgeInfo2.vehiclesStopped
    totalWaitingTime_horizontal = edgeInfo1.waitingTime + edgeInfo2.waitingTime
    totalCO2Emission_horizontal = edgeInfo1.CO2Emission + edgeInfo2.CO2Emission
    numOfCars_horizontal = edgeInfo1.carInfo["Number"] + edgeInfo2.carInfo["Number"]
    numOfBuses_horizontal = edgeInfo1.busInfo["Number"] + edgeInfo2.busInfo["Number"]
    numOfTrucks_horizontal = edgeInfo1.truckInfo["Number"] + edgeInfo2.truckInfo["Number"]
    numOfMotorcycles_horizontal = edgeInfo1.motorcycleInfo["Number"] + edgeInfo2.motorcycleInfo["Number"]
    numOfBicycles_horizontal = edgeInfo1.bicycleInfo["Number"] + edgeInfo2.bicycleInfo["Number"]
    numOfPredictedVehiclesAtTLS_horizontal = edgeInfo1.predictedVehiclesAtTLS["Number"] + edgeInfo2.predictedVehiclesAtTLS["Number"]

    numOfVehiclesStopped_vertical = edgeInfo3.vehiclesStopped + edgeInfo4.vehiclesStopped
    totalWaitingTime_vertical = edgeInfo3.waitingTime + edgeInfo4.waitingTime
    totalCO2Emission_vertical = edgeInfo3.CO2Emission + edgeInfo4.CO2Emission
    numOfCars_vertical = edgeInfo3.carInfo["Number"] + edgeInfo4.carInfo["Number"]
    numOfBuses_vertical = edgeInfo3.busInfo["Number"] + edgeInfo4.busInfo["Number"]
    numOfTrucks_vertical = edgeInfo3.truckInfo["Number"] + edgeInfo4.truckInfo["Number"]
    numOfMotorcycles_vertical = edgeInfo3.motorcycleInfo["Number"] + edgeInfo4.motorcycleInfo["Number"]
    numOfBicycles_vertical = edgeInfo3.bicycleInfo["Number"] + edgeInfo4.bicycleInfo["Number"]
    numOfPredictedVehiclesAtTLS_vertical = edgeInfo3.predictedVehiclesAtTLS["Number"] + edgeInfo4.predictedVehiclesAtTLS["Number"]

    return [numOfVehiclesStopped_horizontal, totalWaitingTime_horizontal, totalCO2Emission_horizontal, numOfPredictedVehiclesAtTLS_horizontal, numOfCars_horizontal, numOfBuses_horizontal, numOfTrucks_horizontal, numOfMotorcycles_horizontal, numOfBicycles_horizontal,
            numOfVehiclesStopped_vertical, totalWaitingTime_vertical, totalCO2Emission_vertical, numOfPredictedVehiclesAtTLS_vertical, numOfCars_vertical, numOfBuses_vertical, numOfTrucks_vertical, numOfMotorcycles_vertical, numOfBicycles_vertical]

def displayCumulative_at_once(dataFrame1, dataFrame2, title):
    y1, y2, y3 = [], [], []
    j1, j2, j3 = 0, 0, 0
    fig = plt.figure()

    for idx in range(len(dataFrame1)):
        j1 += dataFrame1[idx]
        y1.append(j1)

    for idx in range(len(dataFrame2)):
        j2 += dataFrame2[idx]
        y2.append(j2)

    print(y1[-1], y2[-1])
    
    x1 = np.arange(0, len(y1))
    x2 = np.arange(0, len(y2))

    plt.plot(x1, y1, label='Without Any Model', color='red')
    plt.plot(x2, y2, label='Deep Neural Network', color='blue')
    plt.xlabel('Seconds Elapsed')
    plt.ylabel(f'Cumulative {title}')
    plt.legend()
    plt.show()