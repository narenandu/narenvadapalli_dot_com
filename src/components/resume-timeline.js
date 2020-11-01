import React from "react";
import { VerticalTimeline, VerticalTimelineElement }  from 'react-vertical-timeline-component';
import 'react-vertical-timeline-component/style.min.css';
import pdLogo from '../../static/images/pd_logo.png'
import bardelLogo from '../../static/images/bardel_logo.png'
import onAnimationLogo from '../../static/images/on_animation_logo.jpg'
import animalLogicLogo from '../../static/images/animal_logic_logo.jpg'
import dreamWorksLogo from '../../static/images/dreamworks_logo.jpg'
import eaSportsLogo from '../../static/images/ea_sports_logo.jpg'
import samsungLogo from '../../static/images/samsung_logo.jpg'
import motoLogo from '../../static/images/motorola_logo.jpg'
import upennLogo from '../../static/images/upenn_logo.jpg'
import nitwLogo from '../../static/images/nitw_logo.jpg'
import "../style/resume-timeline.less";

export default function(){
    return (
        <div>
        <section id="WorkExperiencePage">
            <h2 class="titles">Work Experience</h2>
            <VerticalTimeline>
                <VerticalTimelineElement
                    className="vertical-timeline-element--work"
                    contentStyle={{ background: 'rgb(26,49,82)', color: '#ffe' }}
                    contentArrowStyle={{ borderRight: '7px solid  rgb(26,49,82)' }}
                    date="2020 (Feb) - Present"
                    iconStyle={{ background: 'rgb(26,49,82)', color: '#fff' }}
                    icon={<img alt=""  class="logo-size" src={pdLogo}></img>}>
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
                    icon={<img alt=""  class="logo-size" src={bardelLogo}></img>}>
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
                    icon={<img alt=""  class="logo-size" src={onAnimationLogo}></img>}>
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
                    icon={<img alt=""  class="logo-size" src={animalLogicLogo}></img>}>
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
                    icon={<img alt=""  class="logo-size" src={dreamWorksLogo}></img>}>
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
                    icon={<img alt=""  class="logo-size" src={eaSportsLogo}></img>}>
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
                    icon={<img alt=""  class="logo-size" src={samsungLogo}></img>}>
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
                    icon={<img alt=""  class="logo-size" src={motoLogo}></img>}>
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
                icon={<img alt=""  class="logo-size" src={upennLogo}></img>}>
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
                icon={<img alt=""  class="logo-size" src={nitwLogo}></img>}>
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
    </div>        
    )
}
