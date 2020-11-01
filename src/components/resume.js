import React from "react";
import SectionTitle from "./sectiontitle";
import { StaticQuery, graphql } from "gatsby";
import SocialLinks from "./sociallinks";
import ResumeTimeLine from "./resume-timeline";

const Resume = function() {
    return (
        <section id="contact" className="container">
            <div className="section-title">
                <SectionTitle title="RESUME" />
            </div>
            <ResumeTimeLine />
            <div className={"row no-form"} >
                <div className={"col s12 details"} >
                    <ul>
                        <li>
                            <SocialLinks />
                        </li>
                    </ul>
                </div>
            </div>

        </section>
    );
}

export default () => (
    <StaticQuery
        query={graphql`
            query {
                site {
                    siteMetadata {
                        contact {
                            api_url
                            description
                            mail
                            phone
                            address
                        }
                    }
                }
            }
        `}
        render={data => <Resume />}
    />
);
