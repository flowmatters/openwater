import requests
import sys
import os
import urllib
import io
import zipfile
import shutil
from glob import glob
from string import Template

ARTIFACTS_URL='https://api.github.com/repos/${org}/${repo}/actions/artifacts'
DEST_RELATIVE='~/.openwater/installations'

def platform():
    p = sys.platform
    if p=='win32':
        return 'windows'
    return p

def get_artifacts(org='flowmatters',repo='openwater-core',limit=None,**kwargs):
    '''
    Retrieve the list of build artifacts currently available

    Parameters:

    org, repo - Github organisation and repository
    limit - Number of most recent artifacts to return
    kwargs - 
        used to match particular artifacts. Can be any field from github API, as well as
        'platform' and 'sha', which we compute based on artifact['name'] based on
        the convention used in openwater-core (artifact name is <platform>-<sha>.zip)
    '''
    url = Template(ARTIFACTS_URL).substitute(org=org,repo=repo)
    resp = requests.get(url).json()
    artifacts = resp['artifacts']
    artifacts = [a for a in artifacts if not a['expired']]
    for a in artifacts:
        components = a['name'].split('-')
        if len(components) != 2:
            continue
        a['platform'] = components[0]
        a['sha'] = components[1]
    for k,v in kwargs.items():
        artifacts = [a for a in artifacts if k in a and a[k]==v]

    artifacts = sorted(artifacts,key=lambda a:a['updated_at'])[::-1]
    if limit is not None:
        artifacts = artifacts[:limit]
    return artifacts

def latest_available_artifact(**kwargs):
    '''
    Return the most recent artifact, *for the current platform*, subject
    to additional query parameters (kwargs)
    '''
    artifacts = get_artifacts(limit=1,platform=platform(),**kwargs)
    if len(artifacts): return artifacts[0]
    return None
    
def _post_install_linux(artifact,dest):
    programs = ['ow-inspect','ow-single','ow-sim','ows-ensemble']
    for p in programs:
        fn = os.path.join(dest,p)
        os.chmod(fn,755)

def _post_install_windows(artifact,dest):
    dlls = os.path.join(dest,'1.8.21','bin','*.dll')
    for dll in glob(dlls):
        shutil.copyfile(dll,os.path.join(dest,os.path.basename(dll)))


POST_INSTALL = {
    'linux':_post_install_linux,
    'windows':_post_install_windows
}

def install_artifact(access_token,artifact=None,dest=None):
    '''
    Install a given artifact (returned from get_artifacts or latest_available_artifact).

    Parameters:

    * access_token - required Github access token.
    * artifact - the artifact to install. Defaults to latest artifact for this platform
    * dest - directory to install the binaries. 
             Defaults to ~/openwater/installations/<artifact-date>-<commit-sha>

    Returns: destination directory used

    Access tokens can be created from your account settings on Github under
      Settings | Developer Settings | Personal Access Tokens
    '''
    if artifact is None:
        artifact = latest_available_artifact()

    assert artifact['platform'] == platform()
    url = artifact['archive_download_url']

    if dest is None:
        dest = os.path.join(os.path.expanduser(DEST_RELATIVE),f"{artifact['updated_at'].replace(':','')}-{artifact['sha']}")

    if not os.path.exists(dest):
        os.makedirs(dest)

        req = urllib.request.Request(url)
        req.add_header('Authorization', f'token {access_token}')
        remotezip = urllib.request.urlopen(req)
        zipinmemory = io.BytesIO(remotezip.read())
        zip = zipfile.ZipFile(zipinmemory)
        zip.extractall(path=dest)

    POST_INSTALL[artifact['platform']](artifact,dest)
    return dest

