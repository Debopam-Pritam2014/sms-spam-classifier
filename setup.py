from setuptools import find_packages, setup
from typing import List

HYPEN_e='-e .'
def get_requirements(filepath:str)->List[str]:
    '''

    :param filepath:
    :return: a list of r equirement packages
    '''
    requirements=[]
    with open(filepath) as file_obj:
        requirements=file_obj.readline()
        requirements=[req.replace("\n","") for req in requirements]
        if HYPEN_e in requirements:
            requirements.remove(HYPEN_e)
    return requirements


setup(

    name='SMS-spam-Classifier',
    version='0.0.1',
    author='Debopam',
    author_email='debopamdeycse19@gmail.com',
    packages=find_packages(),
    install_require=get_requirements('requirements.txt')



)