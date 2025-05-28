import base64
import json
import zlib
from threading import Thread
from datetime import datetime,timedelta
import pandas as pd
import requests
import socket
import time
from backend.logs.logger_setup import setup_logger

# Configure logging
logger = setup_logger('integration', 'integration.log')

class PrescintoIntegrationUtilities:
    # Prodcution Constant

    __VERSION_INFO = "1.0.5-US"
    __REQUEST_TIMEOUT = 600
    __LIST_SERVERS = [
        {
            "PREF": "http://",
            "IP": "20.25.84.200",
            "PORT": 5002,
            "SERVER": "US",
            "BASE_URL": "http://us-papi.prescinto.ai:5002",
            "IS_PUBLIC": True,
        }
    ]

    def __init__(self, token=None, server=None):
        """_summary_

        Args:
            token (_type_, required): API Token. Defaults to None.
            server (str, optional): API Server to connect to ['India', 'US']. Defaults to 'India'.
        """
        self.__BASE_URL = self.__chooseServer(server=server)

        self.__URL_GET_TEMPLATE = f"{self.__BASE_URL}/api/v1/get/template"
        self.__URL_GET_DATA = f"{self.__BASE_URL}/api/v1/get/data"
        self.__URL_GET_DATA_V2 = f"{self.__BASE_URL}/api/v1/get/dataV2"
        self.__URL_GET_CAPACITY_DETAIL = f"{self.__BASE_URL}/api/v1/get/capacity"
        self.__URL_GET_MODULE_CLEANING = f"{self.__BASE_URL}/api/v1/get/moduleCleaning"
        self.__URL_GET_CUSTOM_TEMPLATE = f"{self.__BASE_URL}/api/v1/get/custom/template"
        self.__URL_GET_PROCESS_DATA = f"{self.__BASE_URL}/api/v1/get/processdata"
        self.__URL_GET_ASESSTS_DETAIL = f"{self.__BASE_URL}/api/v1/get/assetMaster"
        self.__URL_GET_FORECAST_DETAIL = f"{self.__BASE_URL}/api/v1/get/forecast"
        self.__URL_GET_PROJECT_LIST = f"{self.__BASE_URL}/api/v1/get/projectList"
        self.__URL_GET_HIERARCHICAL_STRUCTURE = (
            f"{self.__BASE_URL}/api/v1/get/plant/structure"
        )
        self.__URL_GET_ALIAS_STRUCTURE = (
            f"{self.__BASE_URL}/api/v1/get/plant/aliasStructure"
        )
        self.__URL_CREATE_WORKORDER = f"{self.__BASE_URL}/api/v1/create/workorder"

        self.__URL_GET_CYCLE_DATA = f"{self.__BASE_URL}/api/v1/get/getCycleData"
        self.__URL_GET_DAILY_CYCLE_STATS_DATA = f"{self.__BASE_URL}/api/v1/get/getDailyStats"
        self.__URL_GET_PLANT_LAST_TIMESTAMP = f"{self.__BASE_URL}/api/v1/get/LastTimeStamp"

        if token is None:
            self.__auth_token = "b6d25b56-4eb1-11eb-ae93-0242ac130002"
        else:
            self.__auth_token = token

    def __chooseServer(self, server=None):
        if server:
            selectedServer = [
                s
                for s in PrescintoIntegrationUtilities.__LIST_SERVERS
                if s["SERVER"] == server
            ]
            if len(selectedServer) > 0:
                return selectedServer[0]["BASE_URL"]
            else:
                return self.__soundServers()
        else:
            return self.__soundServers()

    def __soundServers(self):
        list_server_ping = [
            self.__pingServer(
                server=s["IP"], port=s["PORT"], timeout=5, base_url=s["BASE_URL"]
            )
            for s in PrescintoIntegrationUtilities.__LIST_SERVERS
            if s.get("IS_PUBLIC")
        ]
        list_server_ping = [s for s in list_server_ping if s["status"]]
        if len(list_server_ping) > 0:
            list_server_ping.sort(key=lambda x: x["p_time"])
            return list_server_ping[0].get("BASE_URL")
        else:
            logger.error("Not able to reach any Server.")

    def __getTimeOut(self, timeout):
        """AI is creating summary for __getTimeOut
        Args:
            timeout ([int|None]): [timout in seconds. if not provide by user then will be None.]
        Returns:
            [int]: Default timeout, If user input nothing for timeout.
        """
        if not timeout:
            timeout = PrescintoIntegrationUtilities.__REQUEST_TIMEOUT
        timeout = (
            timeout if timeout > 0 else PrescintoIntegrationUtilities.__REQUEST_TIMEOUT
        )
        return timeout

    # Create header parameter in every call
    # Return JSON

    def __header(self):
        return {"Content-type": "application/json", "x-api-key": self.__auth_token}

    def __headerzip(self):
        return {"Content-type": "application/zip", "x-api-key": self.__auth_token}

    def __decode(self, obj):
        return base64.b64decode(obj)

    # Convert base64 data to dataframe
    # Input base64 string
    # Return dataframe
    def __encodeConvertToActualValue(self, encodeValue):
        values = zlib.decompress(self.__decode(encodeValue)).decode("ascii")
        return pd.DataFrame.from_dict(json.loads(json.loads(values)))

    # Get the template alias data based on plant name
    # Input plant name
    # Return None when get invalid request otherwise json data
    def __getAliasStructure(self, pName, timeout=None):
        payload = {"pName": pName}
        return self.__post_call(
            self.__URL_GET_ALIAS_STRUCTURE,
            "alias hierarchical structure api",
            payload,
            timeout,
        )

    # Get the template data based on plant name same as portal
    # Input plant name
    # Return None when get invalid request otherwise json data
    def __getStructure(self, pName, timeout=None):
        payload = {"pName": pName}
        return self.__post_call(
            self.__URL_GET_HIERARCHICAL_STRUCTURE,
            "hierarchical structure api",
            payload,
            timeout,
        )

    # Get the template data based on plant name
    # Input plant name
    # Return None when get invalid request otherwise json data

    def __getTemplate(self, pName, isNodeId, timeout=None):
        payload = {"pName": pName, "isNodeId": isNodeId}

        return self.__post_call(
            self.__URL_GET_TEMPLATE, "template api", payload, timeout
        )

    # Get the template data based on plant name
    # Input plant name
    # Return None when get invalid request otherwise json data
    def __getCustomTemplate(self, pName, timeout=None):
        payload = {"pName": pName}
        return self.__post_call(
            self.__URL_GET_CUSTOM_TEMPLATE, "custom template api", payload, timeout
        )

    def getProcessData(self, timeout=None):
        timeout = self.__getTimeOut(timeout)
        response = requests.get(
            self.__URL_GET_PROCESS_DATA,
            headers=self.__headerzip(),
            stream=True,
            timeout=timeout,
        )

    def __post_call(self, apiEndPoint, apiName, payload, timeout=None):
        dataResp = None
        timeout = self.__getTimeOut(timeout)
        try:
            response = requests.post(
                apiEndPoint, json=payload, headers=self.__header(), timeout=timeout
            )
            # extracting data in json format
            if response.status_code == 200:
                dataRespTemp = response.json()
                if dataRespTemp["response"]["status"] == True:
                    reps = dataRespTemp["response"]["result"]
                    if isinstance(reps, dict):
                        dataResp = reps
                    elif isinstance(reps, str):
                        dataResp = json.loads(reps)
                    else:
                        dataResp = reps
            else:
                logger.error("Response status code: {}".format(response.status_code))
                logger.error("Response error in {0}-: {1}".format(apiName, response.text))
        except requests.Timeout as err:
            logger.error("Timeout Error: {}".format(str(err)))
        except Exception as e:
            logger.error("Error in {0}: {1}".format(apiName, str(e)))
        return dataResp

    # Get the plant data based on filter
    # Input plant name, category list, parameter list, device list, start and end date
    # Return None when get invalid request otherwise dataframe
    def __getData(
        self,
        pName,
        catList,
        paramList,
        deviceList=None,
        sDate=None,
        eDate=None,
        granularity="5m",
        condition=None,
        quality=None,
        timeout=None,
    ):
        if granularity is None:
            granularity = "5m"
        dataResp = None
        try:
            # prepare payload
            payload = {
                "pName": pName,
                "categoryList": catList,
                "parameterList": paramList,
                "deviceList": deviceList,
                "sDate": sDate,
                "eDate": eDate,
                "granularity": granularity,
                "condition": condition,
                "quality": quality,
            }
            timeout = self.__getTimeOut(timeout)
            response = requests.post(
                self.__URL_GET_DATA,
                json=payload,
                headers=self.__header(),
                timeout=timeout,
            )
            # extracting data in json format
            if response.status_code == 200:
                resArr = response.text.split("#######")
                resArr.pop()
                dataResp = self.__getNormalDf(pd.DataFrame(), resArr)
                dataResp = self.__dataFilter(dataResp, paramList, deviceList)
                dataResp = self.__changeDfDateTime(dataResp)

            else:
                logger.error("Response status code: {}".format(response.status_code))
                logger.error("Response error: {}".format(response.text))
        except requests.Timeout as err:
            logger.error("Timeout Error: {}".format(str(err)))
        except Exception as e:
            logger.error("Error in GetData api: {}".format(str(e)))

        return dataResp

    def __getDataV2(
        self,
        pName,
        catList,
        paramList,
        deviceList=None,
        sDate=None,
        eDate=None,
        granularity="5m",
        condition=None,
        quality=None,
        fetchInUserTimeZone=False,
        timeout=None,
    ):
        if granularity is None:
            granularity = "5m"
        dataResp = None

        try:

            # prepare payload
            payload = {
                "pName": pName,
                "categoryList": catList,
                "parameterList": paramList,
                "deviceList": deviceList,
                "sDate": sDate,
                "eDate": eDate,
                "granularity": granularity,
                "condition": condition,
                "quality": quality,
                "isUserTimeZone": fetchInUserTimeZone,
            }
            timeout = self.__getTimeOut(timeout)
            response = requests.post(
                self.__URL_GET_DATA_V2,
                json=payload,
                headers=self.__header(),
                timeout=timeout,
            )
            # extracting data in json format
            if response.status_code == 200:
                data = response.json()
                if data["response"]["status"]:
                    resp = data["response"]["result"]
                    dataResp = pd.read_json(resp, orient="split")
                    return dataResp
                else:
                    dataResp = data["response"]["result"]
            else:
                logger.error("Response status code: {}".format(response.status_code))
                logger.error("Response error: {}".format(response.text))
        except requests.Timeout as err:
            logger.error("Timeout Error: {}".format(str(err)))
        except Exception as e:
            logger.error("Error in GetDataV2: {}".format(str(e)))
        return dataResp

    def __changeDfDateTime(self, dataResp):
        if not dataResp.empty:
            dataResp["time"] = dataResp["time"].apply(
                lambda x: x[:-1] if x.endswith("Z") else x
            )
        return dataResp

    def __dataFilter(self, df, paramList, deviceList):
        isFilter = (
            True if paramList[0].find(".") > -1 and deviceList is not None else False
        )
        contains = df.columns[1]
        pName = paramList[0].split(".")[0]
        flag = False
        if contains.find(pName) > -1:
            flag = True
        if isFilter:
            filterColumns = ["time"]
            for idx in range(len(deviceList)):
                for idx1 in range(len(paramList)):
                    if flag:
                        pName = "_" + paramList[idx1]
                    else:
                        pName = "." + paramList[idx1].split(".")[1]

                    filterColumns.append(deviceList[idx] + pName)

            return df[filterColumns]
        else:
            return df

    def __getNormalDf(self, superDf, result):
        for data in result:
            if data == "None":
                continue

            df = pd.DataFrame(json.loads(data))
            if superDf.empty:
                superDf = df
            else:
                superDf = pd.merge(superDf, df, how="left")

        return superDf

    # Get the plant capacity details
    # Input plant name
    # Return None when get invalid request otherwise json data
    def __getCapacityDetails(self, pName, timeout=None):
        payload = {"pName": pName}
        return self.__post_call(
            self.__URL_GET_CAPACITY_DETAIL, "capacity detail api", payload, timeout
        )

    # Get the Cleaning Module data based on plant name
    # Input plant name, start and end date
    # Return None when get invalid request otherwise json data
    def __getCleaningModule(self, pName, sDate, eDate, timeout=None):
        # prepare payload
        payload = {"pName": pName, "sDate": sDate, "eDate": eDate}
        return self.__post_call(
            self.__URL_GET_MODULE_CLEANING, "module cleaning api", payload, timeout
        )

    # Get the plant asesst master details
    # Input plant name
    # Return None when get invalid request otherwise json data

    def __getAsesstMasterDetails(self, pName, timeout=None):
        payload = {"pName": pName}
        resp = self.__post_call(
            self.__URL_GET_ASESSTS_DETAIL, "asesst detail api", payload, timeout
        )
        return pd.DataFrame.from_dict(resp)

    # Get the plant forecast details
    # Input plant name
    # Return None when get invalid request otherwise json data
    def __getForecastDetail(self, pName, sDate, eDate, timeout=None):
        payload = {"pName": pName, "sDate": sDate, "eDate": eDate}
        return self.__post_call(
            self.__URL_GET_FORECAST_DETAIL, "forecast detail api", payload, timeout
        )

    # Get the all plant list
    # Input N/A
    # Return None when get invalid request otherwise json data
    def __getProjectList(self, pName=None,isRefresh=False, timeout=None):
        payload ={}
        if pName:
            payload = {"pName": pName}
        if isRefresh:
            payload['isRefresh']=isRefresh

        return self.__post_call(
            self.__URL_GET_PROJECT_LIST,
            "project list api",
            payload if pName or isRefresh else None,
            timeout,
        )

    # Create the Workorder based on plant name
    # Input plant name, device name, remark
    # Return based on thrid party response
    def __createWorkOrders(self, pName, deviceName, remark, timeout=None):
        # prepare payload
        payload = {"pName": pName, "devName": deviceName, "remark": remark}
        return self.__post_call(
            self.__URL_CREATE_WORKORDER, "create work order api", payload, timeout
        )

    # call all API's
    # Input plant Name, event (based on API call)
    # Return data
    def __apiCall(self, plantName, event, isNodeId, timeout=None):

        if event == "Template":
            return self.__getTemplate(plantName, isNodeId, timeout)
        elif event == "Custom":
            return self.__getCustomTemplate(plantName, timeout)
        elif event == "Structure":
            return self.__getStructure(plantName, timeout)
        elif event == "Alias":
            return self.__getAliasStructure(plantName, timeout)
        else:
            return self.__getCapacityDetails(plantName, timeout)

    # Async api call to improve performace
    # Input plant name, api name, dict (response store in this dict)
    # Return dict
    def __parallelCall(
        self, plantName, apiName, store=None, isNodeId=False, timeout=None
    ):
        if store is None:
            store = {}

        store[apiName] = self.__apiCall(plantName, apiName, isNodeId, timeout)
        return store

    def getPlantStructure(self, plantName, timeout=None):
        callsArr = ["Structure"]
        store = self.__executorService(plantName, callsArr, timeout=timeout)
        data = store["Structure"]
        return data

    def getAliasPlantStructure(self, plantName, timeout=None):

        callsArr = ["Alias"]
        store = self.__executorService(plantName, callsArr, timeout=timeout)
        data = store["Alias"]
        return data

    def getPlantInfo(self, plantName, isNodeId=False, timeout=None):

        callsArr = ["Template"]
        store = self.__executorService(plantName, callsArr, isNodeId, timeout=timeout)

        data = store["Template"]
        catogiers = data["categories"]
        parmDict = {}
        catList = []
        deviceDict = {}

        for idx in range(len(catogiers)):
            for key, name in catogiers[idx].items():
                catList.append(key)
                deviceArr = []
                for idx1 in range(len(name)):
                    if isNodeId:
                        key1, parmArr = self.__getParameterAndNodeKey(
                            name, deviceArr, idx1, None
                        )
                    else:
                        key1, parmArr = self.__getParameterAndKey(
                            name, deviceArr, idx1, None
                        )
                    self.__setParameter(parmDict, key, key1, parmArr)

                deviceDict[key] = deviceArr

        # if weatherList is not None:
        return (catList, deviceDict, parmDict)

    def __setParameter(self, parmDict, key, key1, parmArr):
        if (
            key.lower() == "inverter"
            or key.lower().find("smb") > -1
            or key.upper().find("WS") > -1
            or key.upper().find("WMS") > -1
        ):
            if parmDict.get(key) is None:
                parmDict[key] = parmArr
        else:
            parmDict[key1] = parmArr

    def __getParameterAndNodeKey(self, name, deviceArr, idx1, parmArr):
        for key1, name1 in name[idx1].items():
            dtmp = key1.split("#")
            deviceArr.append(dtmp)
            parmArr = []
            for paramName, paramNodeId in name1.items():
                if paramName not in parmArr:
                    ptmp = []
                    ptmp.append(paramName)
                    ptmp.append(paramNodeId)
                    parmArr.append(ptmp)
        return key1, parmArr

    def __getParameterAndKey(self, name, deviceArr, idx1, parmArr):
        for key1, name1 in name[idx1].items():
            if len(name1) > 0 and name1[0] == "PR":
                continue
            deviceArr.append(key1)
            parmArr = []
            for paramName in name1:
                if paramName not in parmArr:
                    parmArr.append(paramName)
        return key1, parmArr

    def __executorService(self, plantName, callsArr, isNodeId=False, timeout=None):
        store = {}
        threads = []

        # create the threads
        # Parrallel call for both API's
        for i in range(len(callsArr)):
            t = Thread(
                target=self.__parallelCall,
                args=(plantName, callsArr[i], store, isNodeId, timeout),
            )
            threads.append(t)

        # start the threads
        [t.start() for t in threads]

        # wait for the threads to finish
        [t.join() for t in threads]
        return store

    def getWindPlantInfo(self, pName, timeout=None):
        callsArr = ["Custom"]
        store = self.__executorService(pName, callsArr, timeout=timeout)
        data = store["Custom"]
        return (
            data.get("categories"),
            data.get("turbineList"),
            data.get("deviceList"),
            data.get("paramList"),
        )

    def fetchData(
        self,
        pName,
        catList,
        paramList,
        deviceList=None,
        sDate=None,
        eDate=None,
        granularity=None,
        condition=None,
        quality=None,
        timeout=None,
    ):
        dataDf = self.__getData(
            pName,
            catList,
            paramList,
            deviceList,
            sDate,
            eDate,
            granularity,
            condition,
            quality,
            timeout,
        )
        return dataDf

    def fetchDataV2(
        self,
        pName,
        catList,
        paramList,
        deviceList=None,
        sDate=None,
        eDate=None,
        granularity=None,
        condition=None,
        quality=None,
        fetchInUserTimeZone=False,
        timeout=None,
    ):
        """
        Start Date and End Date Format Should be: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' or 'YYYY-MM-DD HH:MM:SS±HH:MM'.
        Here ±HH:MM represents timezone, Always it will start with - or +
        i.e. '2022-10-15' or '2022-10-15 10:15:20' or '2022-10-15 10:15:20+05:30' or '2022-10-15 10:15:20-03:00'

        Args:
        pName (str, required): Plant Short Name.
        catList (str, required): Catgeory List.
        paramList (list, required): Parameters List.
        deviceList (list, required): Devices list, Default to None or Empty List.
        sDate (str, optional): String Start Date. Default to None
        eDate (str, optional):String End Date. Default to None
        granularity (str, optional): Data granularity.Default to 5m i.e. granularity='1m'
        condition (dict, optional): Aggregate Condition for each parameter.i.e. {'State Of Charge':'Min'}. Default to None
        quality (str, optional): Data Quality. Default to Good. Default to None
        fetchInUserTimeZone (bool, optional): Fetch data in user timezone. Default to False
        timeout (int, optional): Request timeout. Default to 600 seconds.

        Returns:
            [DataFrame]: [Multiple Columns, depends on parameter and categories]

        Sample:
                If User want to fetch Data in Plant Timezone

                m.fetchDataV2(pName,catList,paramList,deviceList,sDate ='2022-04-05 10:40:00+05:30',eDate ='2022-04-05 13:43:00+05:30',fetchInUserTimeZone =False)

                If User want to fetch Data in his/her Timezone
                m.fetchDataV2(pName,catList,paramList,deviceList,sDate ='2022-04-05 10:40:00+05:30',eDate ='2022-04-05 13:43:00+05:30',fetchInUserTimeZone =True)

        """

        dataDf = self.__getDataV2(
            pName,
            catList,
            paramList,
            deviceList,
            sDate,
            eDate,
            granularity,
            condition,
            quality,
            fetchInUserTimeZone,
            timeout,
        )
        return dataDf

    def getCapacityDetail(self, pName, timeout=None):
        resCapacity = self.__getCapacityDetails(pName, timeout)
        return resCapacity

    def fetchCleaningData(self, pName, startDate, endDate, timeout=None):
        respModuleCleaning = self.__getCleaningModule(
            pName, startDate, endDate, timeout
        )
        if isinstance(respModuleCleaning, str):
            return respModuleCleaning

        if respModuleCleaning is None:
            return "No record found."

        dfTotalModuleCleaning = pd.DataFrame(respModuleCleaning["ModuleCleaning"][0])
        dfCleanAndPlannedModule = pd.DataFrame(respModuleCleaning["ModuleCleaning"][1])
        return dfTotalModuleCleaning, dfCleanAndPlannedModule

    def getAsesstDetails(self, pName, timeout=None):
        resCapacity = self.__getAsesstMasterDetails(pName, timeout)
        return resCapacity

    def getForecastDetails(self, pName, sDate=None, eDate=None, timeout=None):
        resCapacity = self.__getForecastDetail(pName, sDate, eDate, timeout)
        return pd.DataFrame.from_dict(resCapacity.get("data"))

    def getProjectLists(self, pName=None, timeout=None,isRefresh=False):
        res = self.__getProjectList(pName, timeout =timeout,isRefresh=isRefresh)
        return pd.DataFrame.from_dict(res)

    def createWorkOrder(self, pName, deviceName, remark, timeout=None):
        return self.__createWorkOrders(pName, deviceName, remark, timeout)

    def __postCycleData(
        self, pName, subGroupList, blockList, sDate=None, eDate=None, timeout=None
    ):
        dataResp = None
        try:
            # prepare payload
            payload = {
                "pName": pName,
                "subGroupNames": subGroupList,
                "blockNames": blockList,
                "sDate": sDate,
                "eDate": eDate,
            }
            timeout = self.__getTimeOut(timeout)
            response = requests.post(
                self.__URL_GET_CYCLE_DATA,
                json=payload,
                headers=self.__header(),
                timeout=timeout,
            )
            if response.status_code == 200:
                try:
                    response = response.json()
                    response = response.get("response")
                    if not response.get("status"):
                        logger.error("Error: {}".format(response.get("result")))
                        return dataResp
                except Exception as e:
                    pass

                resArr = response.text.split("#######")
                resArr.pop()
                dataResp = self.__getNormalDf(pd.DataFrame(), resArr)
                return dataResp

            logger.error("Response status code: {}".format(response.status_code))
            logger.error("Response error: {}".format(response.text))
        except Exception as e:
            try:
                response_data = response.json()
                logger.error("Response data: {}".format(response_data))
            except:
                pass
            logger.error("Error: {}".format(str(e)))

        return dataResp

    def getCycleData(
        self, pName, sub_group_list, block_list, sDate, eDate, timeout=None
    ):
        """Fetching CycleData for subgroups and blocks within date range

        Args:
            pName (str, required): Plant Name
            sub_group_list (list, optional): Subgroup List or Name of Subgroup. Default to Empty List
            block_list (list, optional): Block List or BLock Name. Default to Empty List
            sDate (str, required):  Start Date
            eDate (str, required):  End Date
            timeout (int, optional): Request Timeout. Defaults to 600.

        Returns:
            [DataFrame]: Returns Data in DataFrame.
        """

        if isinstance(sub_group_list, str):
            sub_group_list = [sub_group_list]
        if isinstance(block_list, str):
            block_list = [block_list]
        return self.__postCycleData(
            pName, sub_group_list, block_list, sDate, eDate, timeout
        )

    def __postDailyStatsData(
        self, pName, subGroupList, blockList, sDate=None, eDate=None, timeout=None
    ):
        dataResp = None
        try:
            # prepare payload
            payload = {
                "pName": pName,
                "subGroupNames": subGroupList,
                "blockNames": blockList,
                "sDate": sDate,
                "eDate": eDate,
            }
            timeout = self.__getTimeOut(timeout)
            response = requests.post(
                self.__URL_GET_DAILY_CYCLE_STATS_DATA,
                json=payload,
                headers=self.__header(),
                timeout=timeout,
            )
            if response.status_code == 200:
                data = response.json()
                if data["response"]["status"]:
                    resp = data["response"]["result"]
                    dataResp = pd.read_json(resp, orient="split")
                    return dataResp
                else:
                    dataResp = data["response"]["result"]
                    logger.error('Request Failed: {}'.format(dataResp))
            else:
                logger.error("Response error: {}".format(response.text))
        except requests.Timeout as err:
            logger.error("Timeout Error: {}".format(str(err)))
        except Exception as e:
            logger.error("Error in Daily Stats: {}".format(str(e)))

        return dataResp

    def getDailyStatsData(
        self, pName, subGroupList=[], blockList=[], sDate=None, eDate=None, timeout=None
    ):
        """Fetching Daily stats data for subgroups and blocks within date range.
        Args:
            pName (str, required): Plant Name
            sub_group_list (list, optional): Subgroup List or Name of Subgroup. Default to Empty List
            block_list (list, optional): Block List or BLock Name. Default to Empty List
            sDate (str, required):  Start Date
            eDate (str, required):  End Date
            timeout (int, optional): Request Timeout. Defaults to 600.

        Returns:
            [DataFrame]: Returns Data in DataFrame.
        """

        if isinstance(subGroupList, str):
            subGroupList = [subGroupList]
        if isinstance(blockList, str):
            blockList = [blockList]
        if eDate is None or eDate=='':
            eDate = datetime.now().strftime("%Y-%m-%d")

        if sDate is None or sDate=='':
            sDate = (datetime.now()+timedelta(days=-1)).strftime("%Y-%m-%d")

        return self.__postDailyStatsData(
            pName, subGroupList, blockList, sDate, eDate, timeout
        )


    def getLastPlantTimeStampData(
        self,
        pName,
        catList,
        paramList,
        deviceList=None,
        timeout=600,
    ):
        """
        Args:
        pName (str, required): Plant Short Name.
        catList (str, required): Catgeory List.
        paramList (list, required): Parameter List.
        deviceList (list, optional): Device list. Default: None or Empty List.
        timeout (int, optional): Request timeout. Default: 600 seconds.

        Returns:
            [DataFrame]: (N, 3) - Time, Name and Value.

        Sample:
                To fetch last value for given plant, device category, parameter and devices.

                m.getLastPlantTimeStampData(pName,catList,paramList,deviceList)

        """
        dataResp = None
        if isinstance(catList, str):
            catList = [catList]
        if isinstance(paramList, str):
            paramList = [paramList]
        try:
            # prepare payload
            payload = {
                "pName": pName,
                "categoryList": catList,
                "parameterList": paramList,
                "deviceList": deviceList
            }
            timeout = self.__getTimeOut(timeout)
            response = requests.post(
                self.__URL_GET_PLANT_LAST_TIMESTAMP,
                json=payload,
                headers=self.__header(),
                timeout=timeout,
            )
            # extracting data in json format
            if response.status_code == 200:
                data = response.json()
                if data["response"]["status"]:
                    resp = data["response"]["result"]
                    dataResp = pd.read_json(resp, orient="split")
                    return dataResp
                else:
                    logger.error('Failed: {}'.format(data["response"]))
                    dataResp = data["response"]["result"]
            else:
                logger.error("Response status code: {}".format(response.status_code))
                logger.error("Response error: {}".format(response.text))
        except requests.Timeout as err:
            logger.error("Timeout Error: {}".format(str(err)))
        except Exception as e:
            logger.error("Error in LastTimestampValue: {}".format(str(e)))
        return dataResp

    def __pingServer(self, server: str, port: int, base_url: str, timeout=3):
        """ping server"""
        try:
            st = time.time()
            socket.setdefaulttimeout(timeout)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((server, port))
        except OSError as error:
            logger.error(f"Ping Error Ocurred For: {base_url} and Error is: {str(error)}")
            return {"status": False, "p_time": 1000, "BASE_URL": base_url}
        else:
            s.close()
            return {"status": True, "p_time": time.time() - st, "BASE_URL": base_url}

    def getVersion(self):
        return PrescintoIntegrationUtilities.__VERSION_INFO

    def getServer(self):
        return self.__BASE_URL
