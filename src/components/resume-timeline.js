import React from "react";
import { VerticalTimeline, VerticalTimelineElement }  from 'react-vertical-timeline-component';
import 'react-vertical-timeline-component/style.min.css';
import pdLogo from '../../static/images/pd_logo.png'
import bardelLogo from '../../static/images/bardel_logo.png'
import onAnimationLogo from '../../static/images/on_animation_logo.jpg'
import animalLogicLogo from '../../static/images/animal_logic_logo.jpg'
import dreamWorksLogo from '../../static/images/dreamworks_logo.jpg'
import "../style/resume-timeline.less";

export default function(){
    return (
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
                    Worked as a Department Technical Director for FX for the features
                    "Penguins of Madagascar", "Kung Fu Panda 3", "Boss Baby".
                    Day to day duties involve the development of Pipeline tools for FX artists.
                    Debugging issues related to Pipeline flow as well as HOUDINI. Knowing the
                    dependencies with immediate departments Lighting and Animation/Character Effects/Crowds.

                    </p>
                </VerticalTimelineElement>
            </VerticalTimeline>
        </section>
    )
}
