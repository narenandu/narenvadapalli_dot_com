import React from "react";
import { VerticalTimeline, VerticalTimelineElement }  from 'react-vertical-timeline-component';
import 'react-vertical-timeline-component/style.min.css';
import "../style/resume-timeline.less";
import pdLogo from '../../static/images/work_edu/pd_logo.png'
import bardelLogo from '../../static/images/work_edu/bardel_logo.png'
import onAnimationLogo from '../../static/images/work_edu/on_animation_logo.jpg'
import animalLogicLogo from '../../static/images/work_edu/animal_logic_logo.jpg'
import dreamWorksLogo from '../../static/images/work_edu/dreamworks_logo.jpg'
import eaSportsLogo from '../../static/images/work_edu/ea_sports_logo.jpg'
import samsungLogo from '../../static/images/work_edu/samsung_logo.jpg'
import motoLogo from '../../static/images/work_edu/motorola_logo.jpg'
import upennLogo from '../../static/images/work_edu/upenn_logo.jpg'
import nitwLogo from '../../static/images/work_edu/nitw_logo.jpg'
import languagesLogo from '../../static/images/skills/languages.png'
import pythonLogo from '../../static/images/skills/python.png'
import jsLogo from '../../static/images/skills/javascript.svg'
import cLogo from '../../static/images/skills/clang.svg'
import cppLogo from '../../static/images/skills/cplusplus.png'
import goLogo from '../../static/images/skills/golang.svg'
import cloudLogo from '../../static/images/skills/cloud.png'
import awsLogo from '../../static/images/skills/aws.png'
import gcpLogo from '../../static/images/skills/gcp.png'
import dccLogo from '../../static/images/skills/dcc.jpg'
import mayaLogo from '../../static/images/skills/maya.jpg'
import houdiniLogo from '../../static/images/skills/houdini.jpg'
import katanaLogo from '../../static/images/skills/katana.png'
import nukeLogo from '../../static/images/skills/nuke.png'
import oiioLogo from '../../static/images/skills/oiio.png'
import pyqtLogo from '../../static/images/skills/pyqt.png'
import guerillaLogo from '../../static/images/skills/guerilla.png'
import shotgunLogo from '../../static/images/skills/shotgun.png'
import photoshopLogo from '../../static/images/skills/photoshop.png'
import aftereffectsLogo from '../../static/images/skills/aftereffects.png'
import gimpLogo from '../../static/images/skills/gimp.png'
import osLogo from '../../static/images/skills/os.png'
import linuxLogo from '../../static/images/skills/linux.png'
import windowsLogo from '../../static/images/skills/windows.png'
import redhatLogo from '../../static/images/skills/redhat.svg'
import ubuntuLogo from '../../static/images/skills/ubuntu.png'
import fedoraLogo from '../../static/images/skills/fedora.png'
import webFrameWorksLogo from '../../static/images/skills/web_frameworks.png'
import flaskLogo from '../../static/images/skills/flask.png'
import reactLogo from '../../static/images/skills/react.png'
import grpcLogo from '../../static/images/skills/grpc.png'
import jqueryLogo from '../../static/images/skills/jquery.svg'
import databaseLogo from '../../static/images/skills/db.png'
import pandasLogo from '../../static/images/skills/pandas.png'
import numpyLogo from '../../static/images/skills/numpy.png'
import matplotlibLogo from '../../static/images/skills/matplotlib.png'
import mysqlLogo from '../../static/images/skills/mysql.png'
import devOpsLogo from '../../static/images/skills/devops.png'
import dockerLogo from '../../static/images/skills/docker.png'
import openshiftLogo from '../../static/images/skills/openshift.png'
import jenkinsLogo from '../../static/images/skills/jenkins.svg'
import versionControlLogo from '../../static/images/skills/version_control.jpg'
import gitLogo from '../../static/images/skills/git.svg'
import githubLogo from '../../static/images/skills/github.png'
import bitbucketLogo from '../../static/images/skills/bitbucket.jpg'
import gitlabLogo from '../../static/images/skills/gitlab.jpg'
import perforceLogo from '../../static/images/skills/perforce.png'
import productivityLogo from '../../static/images/skills/productivity.jpg'
import confluenceLogo from '../../static/images/skills/confluence.jpg'
import jiraLogo from '../../static/images/skills/jira.jpg'

export default function(){
    return (
        <div>
        <p className="titles">
            I am a Software Developer who believes in SOLID principles and Test Driven Development (TDD)
        </p>
        <section id="WorkExperiencePage">
            <h2 class="titles">Work Experience</h2>
            <VerticalTimeline>
                <VerticalTimelineElement
                    className="vertical-timeline-element--work"
                    contentStyle={{ background: 'rgb(26,49,82)', color: '#ffe' }}
                    contentArrowStyle={{ borderRight: '7px solid  rgb(26,49,82)' }}
                    date="2020 (Feb) - Present"
                    iconStyle={{ background: 'rgb(26,49,82)', color: '#fff' }}
                    icon={<img alt="Parallel Domain" title="Parallel Domain" class="logo-size" src={pdLogo}></img>}>
                    <h6 className="vertical-timeline-element-title">Software Engineer</h6>
                    <h7 className="vertical-timeline-element-subtitle">Parallel Domain, Vancouver, Canada</h7>
                    <p>
                    Python APIs for core modules used in the production workflows.
                    Building internal Pipeline tools like dataset converters. Creating internal
                    web Dashboards written in React.
                    </p>
                </VerticalTimelineElement>
                <VerticalTimelineElement
                    className="vertical-timeline-element--work"
                    contentStyle={{ background: 'rgb(26,49,82)', color: '#ffe' }}
                    contentArrowStyle={{ borderRight: '7px solid  rgb(26,49,82)' }}
                    date="2018 (Jul) - 2020 (Feb)"
                    iconStyle={{ background: 'rgb(26,49,82)', color: '#fff' }}
                    icon={<img alt="Bardel Entertainment Inc" title="Bardel Entertainment Inc" class="logo-size" src={bardelLogo}></img>}>
                    <h6 className="vertical-timeline-element-title">Senior Pipeline Technical Director</h6>
                    <h7 className="vertical-timeline-element-subtitle">Bardel Entertainment Inc, Vancouver, Canada</h7>
                    <p>
                        Member of the core Pipeline Team, contributing to the Design and Development of next generation
                        Software Components in the pipeline, such as custom schema based version control system for
                        data/assets in the Pipeline.
                        Adding Tests, Refactoring the legacy code to bring it to certain level of stability with a
                        CI/CD built around the legacy code using JenkinsCI, Docker, helm and RedHat Openshift
                        Web apps written in python flask and socketio, example an interface between ShotGun and Production
                        Pipeline (Render farm submitter with status updates on submission)
                    </p>
                </VerticalTimelineElement>
                <VerticalTimelineElement
                    className="vertical-timeline-element--work"
                    contentStyle={{ background: 'rgb(26,49,82)', color: '#ffe' }}
                    contentArrowStyle={{ borderRight: '7px solid  rgb(26,49,82)' }}
                    date="2017 (Oct) - 2018 (Jun)"
                    iconStyle={{ background: 'rgb(26,49,82)', color: '#fff' }}
                    icon={<img alt="On Animation Studios" title="On Animation Studios" class="logo-size" src={onAnimationLogo}></img>}>
                    <h6 className="vertical-timeline-element-title">Chief Pipeline Technical Director</h6>
                    <h7 className="vertical-timeline-element-subtitle">On Animation Studios, Montreal, Canada</h7>
                    <p>
                    Supervising the Pipeline Team and Render/Data Wrangling team at the Studio.
                    Overseeing the Pipeline needs across all the departments in the studio.
                    Developing the Core Pipeline Modules to be used across the studio.
                    Front end departments (Previz/Layout/Animation) Toolchain development.
                    Pipeline development from scratch for FX department.
                    Pipeline development for Comp and Lighting departments.
                    Developed Batch Rendering system using Guerilla Renderer with Data analytics in place for better prediction of rendering times and also for future estimation.
                    </p>
                </VerticalTimelineElement>
                <VerticalTimelineElement
                    className="vertical-timeline-element--work"
                    contentStyle={{ background: 'rgb(26,49,82)', color: '#ffe' }}
                    contentArrowStyle={{ borderRight: '7px solid  rgb(26,49,82)' }}
                    date="2016 (Jan) - 2017 (Jul)"
                    iconStyle={{ background: 'rgb(26,49,82)', color: '#fff' }}
                    icon={<img alt="Animal Logic" title="Animal Logic" class="logo-size" src={animalLogicLogo}></img>}>
                    <h6 className="vertical-timeline-element-title">Pipeline Technical Director for Lighting</h6>
                    <h7 className="vertical-timeline-element-subtitle">Animal Logic, Sydney, Australia</h7>
                    <p>
                    Supervising the Pipeline Team and Render/Data Wrangling team at the Studio.
                    Overseeing the Pipeline needs across all the departments in the studio.
                    Developing the Core Pipeline Modules to be used across the studio.
                    Front end departments (Previz/Layout/Animation) Toolchain development.
                    Pipeline development from scratch for FX department.
                    Pipeline development for Comp and Lighting departments.
                    Developed Batch Rendering system using Guerilla Renderer with Data analytics in place for better prediction of rendering times and also for future estimation.
                    </p>
                </VerticalTimelineElement>
                <VerticalTimelineElement
                    className="vertical-timeline-element--work"
                    contentStyle={{ background: 'rgb(26,49,82)', color: '#ffe' }}
                    contentArrowStyle={{ borderRight: '7px solid  rgb(26,49,82)' }}
                    date="2012 (Feb) - 2015 (Dec)"
                    iconStyle={{ background: 'rgb(26,49,82)', color: '#fff' }}
                    icon={<img alt="DreamWorks Animations" title="DreamWorks Animations" class="logo-size" src={dreamWorksLogo}></img>}>
                    <h6 className="vertical-timeline-element-title">Department Technical Director</h6>
                    <h7 className="vertical-timeline-element-subtitle">DreamWorks Animations, Bengaluru, India</h7>
                    <p>
                    Worked for FX in the CG Animation features "Penguins of Madagascar", "Kung Fu Panda 3", "Boss Baby".
                    Day to day duties involve the development of Pipeline tools for FX artists.
                    Debugging issues related to Pipeline flow as well as HOUDINI. Knowing the
                    dependencies with immediate departments Lighting and Animation/Character Effects/Crowds.

                    Worked for Animation in CG Animation features like Mr.Peabody and Sherman as an Animation Technical Director.
                    Typical duties involves Animation pipeline maintenance, upgrading/creating new features in the pipeline,
                    Writing scripts, Helping out artists in debugging on the production issues. Past projects include
                    DreamWorks Dragons Riders of Berk, the TV series for Cartoon Network (Episodes: 2,4,10 and 12)
                    </p>
                </VerticalTimelineElement>
                <VerticalTimelineElement
                    className="vertical-timeline-element--work"
                    contentStyle={{ background: 'rgb(26,49,82)', color: '#ffe' }}
                    contentArrowStyle={{ borderRight: '7px solid  rgb(26,49,82)' }}
                    date="2011 (May) - 2011 (Aug)"
                    iconStyle={{ background: 'rgb(26,49,82)', color: '#fff' }}
                    icon={<img alt="EA Sports" title="EA Sports" class="logo-size" src={eaSportsLogo}></img>}>
                    <h6 className="vertical-timeline-element-title">CG Software Engineer - Intern</h6>
                    <h7 className="vertical-timeline-element-subtitle">EA Sports Tiburon, Orlando, FL, USA</h7>
                    <p>
                    Graphics Solutions, Internal Tools and Pipelines for Core Football Graphics Team.
                    This was a 3 months internship where I got a taste of how the game development happens
                    but it was too short a time to do any valuable contribution except for a couple of small bug fixes
                    that went in to Madden/NCAA 2012
                    </p>
                </VerticalTimelineElement>
                <VerticalTimelineElement
                    className="vertical-timeline-element--work"
                    contentStyle={{ background: 'rgb(26,49,82)', color: '#ffe' }}
                    contentArrowStyle={{ borderRight: '7px solid  rgb(26,49,82)' }}
                    date="2007 (Sep) - 2010 (Jul)"
                    iconStyle={{ background: 'rgb(26,49,82)', color: '#fff' }}
                    icon={<img alt="Samsung" title="Samsung" class="logo-size" src={samsungLogo}></img>}>
                    <h6 className="vertical-timeline-element-title">Lead Software Engineer</h6>
                    <h7 className="vertical-timeline-element-subtitle">Samsung India Software Operations, Bengaluru, India</h7>
                    <p>
                    Worked in Embedded software team which develops CDMA mobile applications for USA
                    Tier-2 CDMA carriers. Responsible for the development and maintainance of MP3 Player
                    application front end. Performed several IOTs in messaging and browser modules.
                    </p>
                </VerticalTimelineElement>
                <VerticalTimelineElement
                    className="vertical-timeline-element--work"
                    contentStyle={{ background: 'rgb(26,49,82)', color: '#ffe' }}
                    contentArrowStyle={{ borderRight: '7px solid  rgb(26,49,82)' }}
                    date="2006 (May) - 2007 (Sep)"
                    iconStyle={{ background: 'rgb(26,49,82)', color: '#fff' }}
                    icon={<img alt="Motorola" title="Motorola" class="logo-size" src={motoLogo}></img>}>
                    <h6 className="vertical-timeline-element-title">Software Engineer</h6>
                    <h7 className="vertical-timeline-element-subtitle">Motorola India Pvt Ltd, Bengaluru, India</h7>
                    <p>
                    Responsible for the performance and bug fixing of the BREW PEK module.
                    Validation and fixing of all the BREW APIs as per the carrier specific requirements
                    and device data sheet submitted to Qualcomm (CDMA mobile phones).
                    </p>
                </VerticalTimelineElement>
            </VerticalTimeline>
        </section>
        <section id="EducationPage">
        <h2 class="titles">Education</h2>
        <VerticalTimeline>
            <VerticalTimelineElement
                className="vertical-timeline-element--education"
                contentStyle={{ background: 'rgb(26,49,82)', color: '#ffe' }}
                contentArrowStyle={{ borderRight: '7px solid  rgb(26,49,82)' }}
                date="2010 (Sep) - 2011 (Dec)"
                iconStyle={{ background: 'rgb(26,49,82)', color: '#fff' }}
                icon={<img alt="University of Pennsylvania" title="University of Pennsylvania" class="logo-size" src={upennLogo}></img>}>
                <h6 className="vertical-timeline-element-title">Computer Graphics and Game Technology (MSE)</h6>
                <h7 className="vertical-timeline-element-subtitle">University of Pennsylvania, Philadelphia, PA, USA</h7>
                <p>
                State-of-the-art graphics and animation technologies, as well as interactive media design
                principles, product development methodologies and engineering entrepreneurship
                The CGGT program prepares students for positions requiring multidisciplinary skills such as
                designers, technical animators, technical directors and game programmers
                </p>
            </VerticalTimelineElement>
            <VerticalTimelineElement
                className="vertical-timeline-element--education"
                contentStyle={{ background: 'rgb(26,49,82)', color: '#ffe' }}
                contentArrowStyle={{ borderRight: '7px solid  rgb(26,49,82)' }}
                date="2002 (Sep) - 2006 (May)"
                iconStyle={{ background: 'rgb(26,49,82)', color: '#fff' }}
                icon={<img alt="NIT Warangal" title="NIT Warangal" class="logo-size" src={nitwLogo}></img>}>
                <h6 className="vertical-timeline-element-title">Electronics and Communcation Engineering (BTech)</h6>
                <h7 className="vertical-timeline-element-subtitle">National Institute of Technology, Warangal, India</h7>
                <p>
                The ECE Department at NITW has been an international reputation of excellence in teaching,
                research and service. With excellent laboratory facilities and dedicated faculty, the department
                of ECE offers broad range of programs includes VLSI design.
                </p>
            </VerticalTimelineElement>
        </VerticalTimeline>
        </section>
        <section id="SkillsPage">
        <h2 class="titles">Skills</h2>
        <VerticalTimeline>
            <VerticalTimelineElement
                className="vertical-timeline-element--education"
                contentStyle={{ background: 'rgb(26,49,82)', color: '#ffe' }}
                contentArrowStyle={{ borderRight: '7px solid  rgb(26,49,82)' }}
                iconStyle={{ background: 'rgb(26,49,82)', color: '#fff' }}
                icon={<img alt=""  class="logo-size" src={languagesLogo}></img>}>
                <h6 className="vertical-timeline-element-title">Programming Languages</h6>
                <p>
                    <img alt="python" title="python" class="logo-size" src={pythonLogo}></img>
                    <img alt="javascript" title="javascript" class="logo-size" src={jsLogo}></img>
                    <img alt="C" title="C" class="logo-size" src={cLogo}></img>
                    <img alt="C++" title="C++" class="logo-size" src={cppLogo}></img>
                    <img alt="Go" title="Go" class="logo-size" src={goLogo}></img>
                </p>
            </VerticalTimelineElement>
            <VerticalTimelineElement
                className="vertical-timeline-element--education"
                contentStyle={{ background: 'rgb(26,49,82)', color: '#ffe' }}
                contentArrowStyle={{ borderRight: '7px solid  rgb(26,49,82)' }}
                iconStyle={{ background: 'rgb(26,49,82)', color: '#fff' }}
                icon={<img alt=""  class="logo-size" src={cloudLogo}></img>}>
                <h6 className="vertical-timeline-element-title">Cloud Platforms</h6>
                <p>
                    <img alt="Amazon Web Services=" title="Amazon Web Services=" class="logo-size" src={awsLogo}></img>
                    <img alt="Google Cloud Platform" title="Google Cloud Platform" class="logo-size" src={gcpLogo}></img>
                </p>
            </VerticalTimelineElement>
            <VerticalTimelineElement
                className="vertical-timeline-element--education"
                contentStyle={{ background: 'rgb(26,49,82)', color: '#ffe' }}
                contentArrowStyle={{ borderRight: '7px solid  rgb(26,49,82)' }}
                iconStyle={{ background: 'rgb(26,49,82)', color: '#fff' }}
                icon={<img alt=""  class="logo-size" src={dccLogo}></img>}>
                <h6 className="vertical-timeline-element-title">Digital Content Creation (DCC)</h6>
                <p>
                    <img alt="maya" title="maya" class="logo-size" src={mayaLogo}></img>
                    <img alt="houdini" title="houdini" class="logo-size" src={houdiniLogo}></img>
                    <img alt="katana" title="katana" class="logo-size" src={katanaLogo}></img>
                    <img alt="guerilla" title="guerilla" class="logo-size" src={guerillaLogo}></img>
                    <img alt="nuke" title="nuke" class="logo-size" src={nukeLogo}></img>
                    <img alt="pyqt" title="pyqt" class="logo-size" src={pyqtLogo}></img>
                    <img alt="OpenImageIO" title="openImageIO" class="logo-size" src={oiioLogo}></img>
                    <img alt="shotgun" title="shotgun" class="logo-size" src={shotgunLogo}></img>
                    <img alt="photoshop" title="photoshop" class="logo-size" src={photoshopLogo}></img>
                    <img alt="aftereffects" title="aftereffects" class="logo-size" src={aftereffectsLogo}></img>
                    <img alt="gimp" title="gimp" class="logo-size" src={gimpLogo}></img>
                </p>
            </VerticalTimelineElement>
            <VerticalTimelineElement
                className="vertical-timeline-element--education"
                contentStyle={{ background: 'rgb(26,49,82)', color: '#ffe' }}
                contentArrowStyle={{ borderRight: '7px solid  rgb(26,49,82)' }}
                iconStyle={{ background: 'rgb(26,49,82)', color: '#fff' }}
                icon={<img alt=""  class="logo-size" src={osLogo}></img>}>
                <h6 className="vertical-timeline-element-title">Operating Systems</h6>
                <p>
                    <img alt="linux" title="linux" class="logo-size" src={linuxLogo}></img>
                    <img alt="windows" title="windows" class="logo-size" src={windowsLogo}></img>
                    <img alt="redhat" title="redhat" class="logo-size" src={redhatLogo}></img>
                    <img alt="ubuntu" title="ubuntu" class="logo-size" src={ubuntuLogo}></img>
                    <img alt="fedora" title="fedora" class="logo-size" src={fedoraLogo}></img>
                </p>
            </VerticalTimelineElement>
            <VerticalTimelineElement
                className="vertical-timeline-element--education"
                contentStyle={{ background: 'rgb(26,49,82)', color: '#ffe' }}
                contentArrowStyle={{ borderRight: '7px solid  rgb(26,49,82)' }}
                iconStyle={{ background: 'rgb(26,49,82)', color: '#fff' }}
                icon={<img alt="" class="logo-size" src={webFrameWorksLogo}></img>}>
                <h6 className="vertical-timeline-element-title">Web / Networking</h6>
                <p>
                    <img alt="flask" title="flask" class="logo-size" src={flaskLogo}></img>
                    <img alt="react" title="react" class="logo-size" src={reactLogo}></img>
                    <img alt="grpc" title="grpc" class="logo-size" src={grpcLogo}></img>
                    <img alt="jquery" title="jquery" class="logo-size" src={jqueryLogo}></img>
                </p>
            </VerticalTimelineElement>
            <VerticalTimelineElement
                className="vertical-timeline-element--education"
                contentStyle={{ background: 'rgb(26,49,82)', color: '#ffe' }}
                contentArrowStyle={{ borderRight: '7px solid  rgb(26,49,82)' }}
                iconStyle={{ background: 'rgb(26,49,82)', color: '#fff' }}
                icon={<img alt="" class="logo-size" src={databaseLogo}></img>}>
                <h6 className="vertical-timeline-element-title">Database / DataScience</h6>
                <p>
                    <img alt="mysql" title="mysql" class="logo-size" src={mysqlLogo}></img>
                    <img alt="pandas" title="pandas" class="logo-size" src={pandasLogo}></img>
                    <img alt="numpy" title="numpy" class="logo-size" src={numpyLogo}></img>
                    <img alt="matplotlib" title="matplotlib" class="logo-size" src={matplotlibLogo}></img>
                </p>
            </VerticalTimelineElement>
            <VerticalTimelineElement
                className="vertical-timeline-element--education"
                contentStyle={{ background: 'rgb(26,49,82)', color: '#ffe' }}
                contentArrowStyle={{ borderRight: '7px solid  rgb(26,49,82)' }}
                iconStyle={{ background: 'rgb(26,49,82)', color: '#fff' }}
                icon={<img alt="" class="logo-size" src={devOpsLogo}></img>}>
                <h6 className="vertical-timeline-element-title">DevOps</h6>
                <p>
                    <img alt="docker" title="docker" class="logo-size" src={dockerLogo}></img>
                    <img alt="openshift" title="openshift" class="logo-size" src={openshiftLogo}></img>
                    <img alt="jenkins" title="jenkins" class="logo-size" src={jenkinsLogo}></img>
                </p>
            </VerticalTimelineElement>
            <VerticalTimelineElement
                className="vertical-timeline-element--education"
                contentStyle={{ background: 'rgb(26,49,82)', color: '#ffe' }}
                contentArrowStyle={{ borderRight: '7px solid  rgb(26,49,82)' }}
                iconStyle={{ background: 'rgb(26,49,82)', color: '#fff' }}
                icon={<img alt="" class="logo-size" src={versionControlLogo}></img>}>
                <h6 className="vertical-timeline-element-title">Version Control</h6>
                <p>
                    <img alt="git" title="git" class="logo-size" src={gitLogo}></img>
                    <img alt="github" title="github" class="logo-size" src={githubLogo}></img>
                    <img alt="bitbucket" title="bitbucket" class="logo-size" src={bitbucketLogo}></img>
                    <img alt="gitlab" title="gitlab" class="logo-size" src={gitlabLogo}></img>
                    <img alt="perforce" title="perforce" class="logo-size" src={perforceLogo}></img>
                </p>
            </VerticalTimelineElement>
            <VerticalTimelineElement
                className="vertical-timeline-element--education"
                contentStyle={{ background: 'rgb(26,49,82)', color: '#ffe' }}
                contentArrowStyle={{ borderRight: '7px solid  rgb(26,49,82)' }}
                iconStyle={{ background: 'rgb(26,49,82)', color: '#fff' }}
                icon={<img alt="" class="logo-size" src={productivityLogo}></img>}>
                <h6 className="vertical-timeline-element-title">Productivity / Tracking</h6>
                <p>
                    <img alt="jira" title="jira" class="logo-size" src={jiraLogo}></img>
                    <img alt="confluence" title="confluence" class="logo-size" src={confluenceLogo}></img>
                </p>
            </VerticalTimelineElement>
        </VerticalTimeline>
        </section>
    </div>
    )
}
