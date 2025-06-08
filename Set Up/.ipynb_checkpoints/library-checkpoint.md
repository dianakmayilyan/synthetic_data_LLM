# General-purpose Library Installation
Follow these steps to install general-purpose libraries in your project environment
1. **Verify Package Availability:**
Before proceeding, ensure that thedesired package is available. You can do this by checking the package list provided [here](https://console.cloud.google.com/artifacts/python/vf-grp-aib-prd-mirror/europe-west1/pypi-repository?project=vf-grp-aib-prd-mirror)
2. **Launch a Vertex AI Workbech:**
Once you have verified the package, start a workbech in Vertex AI to begin the installation process
3. **Run the Installation Script:**
Execute the installation script shown below (full script can be found here [here](https://confluence.sp.vodafone.com/pages/viewpage.action?spaceKey=DSG&title=New+Version+of+start-up+script))
<img src="pyproject toml.png" height=300, width=1000>

   *Note:* If you don't have the pyproject.toml file in the target directory, follow these steps to create one
   - Navigate the terminal to the target directory
   - Create the pyproject file by running the following command
   
<img src="touch pyproject.png" height=20, width=200>
   - Open the newly created py project.toml file and copy the necessary lines from this link (https://github.vodafone.com/VFDE-CloudAnalytics/deep-detractor-models/blob/develop/dsl/pyproject.toml)
    
*Note:* in most cases, you'll need the following (replace *"your_project_name"* with your desired name) lines:
<img src="script.png" height=400, width=400>


4. **Create a Virtual Environment:** Run the following command in the terminal to create a virtual environment using Poetry
    
    <img src="poetry shell.png" height=22, width=120>
5. **Install the Libraries:**
With the virtual environment activated, you can now install the desired libraries. (For example 'pip install torch')



         
    
