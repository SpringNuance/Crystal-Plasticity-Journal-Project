line = "tausat_slip               1.05 # per family \n"

lineCoeffPH = "interaction_slipslip    1 1 1 1 1 1 \n"
lineCoeffDB = "interactionSlipSlip 1 1 1 1 1 1  # Interaction coefficients \n"
def replaceAllNumbersInLine(line, num):
    splitLine = line.split(" ")  
    for i in range(len(splitLine)):
        if splitLine[i].isnumeric(): 
            splitLine[i] = str(num)
        elif splitLine[i].endswith("e8"):
            splitLine[i] = str(num) + "e8"
        elif "." in splitLine[i] and splitLine[i].replace(".","").isnumeric():
            splitLine[i] = str(num)
    lineRebuilt = ""
    for word in splitLine:
        lineRebuilt += word + " "
    return lineRebuilt

nums = [2, 3, 4, 5, 6, 7]
def replaceInteractionCoeffs(line, nums):
    splitLine = line.split(" ")  
    counter = 0
    for i in range(len(splitLine)):
        if splitLine[i].isnumeric(): 
            splitLine[i] = str(nums[counter])
            counter += 1
    lineRebuilt = ""
    for word in splitLine:
        lineRebuilt += word + " "
    return lineRebuilt

lineRebuilt = replaceInteractionCoeffs(lineCoeffDB, nums)
print(lineRebuilt)

lineRebuilt = replaceAllNumbersInLine(line, 0.06)
#print(lineRebuilt)
#print("1.05".isnumeric())
#print("1.95".replace(".",""))

nonconverging = [10, 20, 30]
string = ','.join(str(e) for e in nonconverging)
#print(string)
nonconverging = [10, 20]
string = '-'.join(str(e) for e in nonconverging)
#print(string)